// llama-system-one -- ask a System One model typed questions and print the answers.
//
// A System One model does not write text. Give it a state and a set of typed questions and
// it returns, in one forward pass, a distribution over each question's declared options:
//
//   llama-system-one -m model.gguf --state "I'd like two burgers, hold the drink" \
//       --noul   "done:Has the customer finished ordering?" \
//       --choice "item:What is the main item?:burger=a burger,fries=fries,drink=a drink" \
//       --score  "mood:How satisfied do they sound?:angry,unhappy,neutral,happy,delighted"
//
// The same request can be given as the JSON body the server's /v1/systemone route takes
// (--system-one-request), which is the way to keep a prompt identical between the two front ends.
//
// Everything that depends on the checkpoint's readout -- how many sequences to run, which
// tokens carry the answer, where the answer sits -- is decided by the system-one library,
// not here. This file is an adapter: parse arguments, run what the plan asks for, print.

#include "system-one.h"

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static void print_usage(int, char ** argv) {
    LOG("\nexample:\n");
    LOG("  %s -m model.gguf --state \"...\" --noul \"ok:Is the order complete?\"\n", argv[0]);
    LOG("  %s -m model.gguf --system-one-request request.json --json\n", argv[0]);
    LOG("\n");
}

static bool read_file(const std::string & path, std::string & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// "key:instructions:rest" -- the instructions may not contain a colon, which is why the
// JSON body exists for anything more elaborate than a one-off from the shell.
static bool split_spec(const std::string & spec, std::string & key, std::string & instr, std::string & rest) {
    const auto a = spec.find(':');
    if (a == std::string::npos) return false;
    key = spec.substr(0, a);
    const auto b = spec.find(':', a + 1);
    if (b == std::string::npos) {
        instr = spec.substr(a + 1);
        rest.clear();
    } else {
        instr = spec.substr(a + 1, b - a - 1);
        rest  = spec.substr(b + 1);
    }
    return !key.empty();
}

static void parse_options(const std::string & rest, bool with_descs,
                          std::vector<std::string> & options, std::vector<std::string> & descs) {
    for (auto & piece : string_split(rest, ",")) {
        if (piece.empty()) continue;
        const auto eq = with_descs ? piece.find('=') : std::string::npos;
        if (eq == std::string::npos) {
            options.push_back(piece);
            descs.push_back("");
        } else {
            options.push_back(piece.substr(0, eq));
            descs.push_back(piece.substr(eq + 1));
        }
    }
}

// The flags and the JSON body describe the same thing; both land here.
static bool collect_questions(const common_params & params,
                              std::vector<std::string> & keys,
                              std::vector<system_one::question> & qs,
                              std::string & state,
                              std::string & err) {
    if (!params.so_request_file.empty()) {
        std::string body;
        if (!read_file(params.so_request_file, body)) { err = "cannot read " + params.so_request_file; return false; }
        json req;
        try {
            req = json::parse(body);
        } catch (const std::exception & e) {
            err = std::string("invalid request JSON: ") + e.what();
            return false;
        }
        if (req.contains("state")) {
            state = req.at("state").is_string() ? req.at("state").get<std::string>() : req.at("state").dump();
        }
        if (!req.contains("questions") || !req.at("questions").is_object()) {
            err = "request has no \"questions\" object";
            return false;
        }
        for (const auto & item : req.at("questions").items()) {
            const auto & q = item.value();
            const std::string type = q.value("type", "noul");
            system_one::question out;
            out.text = q.value("instructions", "");
            if (type == "noul") {
                out.k = system_one::kind::noul;
            } else if (type == "choice") {
                out.k = system_one::kind::choice;
                if (!q.contains("criteria") || !q.at("criteria").is_object()) {
                    err = "choice question \"" + item.key() + "\" needs a \"criteria\" object";
                    return false;
                }
                for (const auto & c : q.at("criteria").items()) {
                    out.options.push_back(c.key());
                    out.descs.push_back(c.value().is_string() ? c.value().get<std::string>() : std::string());
                }
            } else if (type == "score") {
                out.k = system_one::kind::score;
                if (!q.contains("criteria") || !q.at("criteria").is_array()) {
                    err = "score question \"" + item.key() + "\" needs a \"criteria\" array of levels";
                    return false;
                }
                out.options = q.at("criteria").get<std::vector<std::string>>();
            } else {
                err = "unknown question type \"" + type + "\" for \"" + item.key() + "\"";
                return false;
            }
            keys.push_back(item.key());
            qs.push_back(std::move(out));
        }
        return true;
    }

    for (const auto & spec : params.so_noul) {
        std::string key, instr, rest;
        if (!split_spec(spec, key, instr, rest)) { err = "--noul wants KEY:INSTRUCTIONS"; return false; }
        system_one::question q;
        q.k = system_one::kind::noul;
        q.text = instr;
        keys.push_back(key);
        qs.push_back(std::move(q));
    }
    for (const auto & spec : params.so_choice) {
        std::string key, instr, rest;
        if (!split_spec(spec, key, instr, rest) || rest.empty()) {
            err = "--choice wants KEY:INSTRUCTIONS:opt[=desc][,opt[=desc]...]";
            return false;
        }
        system_one::question q;
        q.k = system_one::kind::choice;
        q.text = instr;
        parse_options(rest, true, q.options, q.descs);
        keys.push_back(key);
        qs.push_back(std::move(q));
    }
    for (const auto & spec : params.so_score) {
        std::string key, instr, rest;
        if (!split_spec(spec, key, instr, rest) || rest.empty()) {
            err = "--score wants KEY:INSTRUCTIONS:level[,level...]";
            return false;
        }
        system_one::question q;
        q.k = system_one::kind::score;
        q.text = instr;
        parse_options(rest, false, q.options, q.descs);
        keys.push_back(key);
        qs.push_back(std::move(q));
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SYSTEM_ONE, print_usage)) {
        return 1;
    }
    common_init();

    // The checkpoint decides the readout, and the readout decides how the context has to be
    // built, so the model is loaded first and the context after -- the same order the server
    // asks its capability questions in.
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == nullptr) {
        LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
        return 1;
    }

    system_one::so_config cfg;
    std::string err;
    if (!system_one::so_config::from_model(model, cfg, err)) {
        LOG_ERR("%s: %s\n", __func__, err.c_str());
        llama_model_free(model);
        return 1;
    }
    if (!params.so_template_file.empty()) {
        if (!read_file(params.so_template_file, cfg.template_src)) {
            LOG_ERR("%s: cannot read template %s\n", __func__, params.so_template_file.c_str());
            llama_model_free(model);
            return 1;
        }
    }

    std::string state = params.so_state;
    if (!params.so_state_file.empty() && !read_file(params.so_state_file, state)) {
        LOG_ERR("%s: cannot read state file %s\n", __func__, params.so_state_file.c_str());
        llama_model_free(model);
        return 1;
    }

    std::vector<std::string>           keys;
    std::vector<system_one::question>  qs;
    if (!collect_questions(params, keys, qs, state, err)) {
        LOG_ERR("%s: %s\n", __func__, err.c_str());
        llama_model_free(model);
        return 1;
    }
    if (qs.empty()) {
        LOG_ERR("%s: no questions -- pass --noul / --choice / --score, or --system-one-request FILE\n", __func__);
        llama_model_free(model);
        return 1;
    }

    system_one::plan plan;
    if (!system_one::build_plan(cfg, llama_model_get_vocab(model), state, qs, plan, err)) {
        LOG_ERR("%s: %s\n", __func__, err.c_str());
        llama_model_free(model);
        return 1;
    }

    // What the checkpoint's readout needs of the context, applied here rather than by the
    // library, so every value the context is built with is visible at this one place.
    const system_one::context_needs need = system_one::required_context(cfg, plan);

    if (need.embeddings) {
        params.embedding = true;
    }
    if (need.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED) {
        params.pooling_type = need.pooling_type;
    }
    if (need.n_seq > 1) {
        params.n_parallel = std::max<int32_t>(params.n_parallel, (int32_t) need.n_seq);
        params.n_ubatch   = std::max<int32_t>(params.n_ubatch,   (int32_t) need.n_tokens);
        params.n_batch    = std::max<int32_t>(params.n_batch,    params.n_ubatch);
        params.n_ctx      = std::max<int32_t>(params.n_ctx,      params.n_ubatch);
    }

    llama_context_params cparams = common_context_params_to_llama(params);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        LOG_ERR("%s: failed to create the context\n", __func__);
        llama_model_free(model);
        return 1;
    }

    std::vector<system_one::answer> answers;
    if (!system_one::run_plan(ctx, plan, answers, err)) {
        LOG_ERR("%s: %s\n", __func__, err.c_str());
        llama_free(ctx); llama_model_free(model);
        return 1;
    }

    // One rendering loop for every readout: the answers arrived in request order either way.
    json out_answers;
    int32_t n_tokens = 0;
    for (const auto & s : plan.sequences) n_tokens += (int32_t) s.tok.ids.size();

    for (size_t qi = 0; qi < qs.size(); qi++) {
        const auto & a = answers[qi];
        const auto & q = qs[qi];
        json ja;
        switch (q.k) {
            case system_one::kind::noul:
                ja["noul"] = a.probs.size() > 1 ? a.probs[1] : 0.0f;
                break;
            case system_one::kind::choice: {
                ja["choice"] = q.options[a.choice];
                json probs;
                for (size_t i = 0; i < q.options.size() && i < a.probs.size(); i++) {
                    probs[q.options[i]] = a.probs[i];
                }
                ja["probabilities"] = probs;
            } break;
            case system_one::kind::score:
                ja["score"]         = system_one::score_expectation(a);
                ja["legend"]        = q.options;
                ja["probabilities"] = a.probs;
                break;
        }
        ja["confidence"] = a.confidence;
        ja["logits"]     = a.logits;
        out_answers[keys[qi]] = ja;
    }

    if (params.so_json) {
        json res;
        res["model"]   = params.model.path;
        res["answers"] = out_answers;
        res["usage"]   = json{
            {"input_tokens",  n_tokens},
            {"output_tokens", 0},
            {"sequences",     (int) plan.sequences.size()},
        };
        LOG("%s\n", res.dump(2).c_str());
    } else {
        for (size_t qi = 0; qi < qs.size(); qi++) {
            const auto & a = answers[qi];
            const auto & q = qs[qi];
            LOG("\n%s  (%s, confidence %.3f)\n", keys[qi].c_str(),
                q.k == system_one::kind::noul   ? "noul"   :
                q.k == system_one::kind::choice ? "choice" : "score", a.confidence);
            switch (q.k) {
                case system_one::kind::noul:
                    LOG("  P(true) = %.4f\n", a.probs.size() > 1 ? a.probs[1] : 0.0f);
                    break;
                case system_one::kind::choice:
                    LOG("  -> %s\n", q.options[a.choice].c_str());
                    for (size_t i = 0; i < q.options.size() && i < a.probs.size(); i++) {
                        LOG("     %-24s %.4f\n", q.options[i].c_str(), a.probs[i]);
                    }
                    break;
                case system_one::kind::score:
                    LOG("  score = %.4f over %zu levels\n", system_one::score_expectation(a), q.options.size());
                    for (size_t i = 0; i < q.options.size() && i < a.probs.size(); i++) {
                        LOG("     %-24s %.4f\n", q.options[i].c_str(), a.probs[i]);
                    }
                    break;
            }
        }
        LOG("\n%d prompt tokens, %zu sequence(s), 0 tokens generated\n",
            n_tokens, plan.sequences.size());
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
