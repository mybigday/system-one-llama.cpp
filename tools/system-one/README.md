# llama-system-one

Ask a System One model typed questions. It does not write text: given a state and a set of
questions, one forward pass returns a distribution over each question's declared options.

```bash
llama-system-one -m model.gguf \
    --state "Customer: two cheeseburgers and a large fries. Cancel the drink." \
    --noul   "done:Has the customer finished ordering?" \
    --choice "item:What is the main item?:burger=a burger,fries=fries,drink=a drink" \
    --score  "mood:How satisfied do they sound?:angry,unhappy,neutral,happy,delighted"
```

```
done  (noul, confidence 0.275)
  P(true) = 0.7985

item  (choice, confidence 0.148)
  -> fries
     burger                   0.2596
     fries                    0.5971
     drink                    0.1433
```

| option | |
|---|---|
| `--state TEXT` / `--state-file FNAME` | the context the questions are asked about |
| `--noul KEY:INSTRUCTIONS` | a true/false question; repeatable |
| `--choice KEY:INSTRUCTIONS:opt[=desc][,...]` | pick one declared option; repeatable |
| `--score KEY:INSTRUCTIONS:level[,...]` | ordered levels, and the expectation over them |
| `--system-one-request FNAME` | the same JSON body `/v1/systemone` takes, instead of the flags |
| `--system-one-template-file FNAME` | override the checkpoint's System One template |
| `--labels A,B,C` | override the labels an answer is read at (default A-Za-z) |
| `--system-one-temperature T` | rescale every answer's logits by T (default 1, skipped) |
| `--json` | print what the server route would return |

The prompt format belongs to the checkpoint, not to this tool: it is a Jinja template stored
as `tokenizer.chat_template.system_one`, and a model without one is an error rather than a
guess. `--system-one-template-file` is the escape hatch. This is **not** the chat template:
a System One template is rendered with a state, questions and option labels, never with a
message list, which is why it does not reuse `--chat-template-file`.

`llama-server` exposes the same thing over HTTP as `POST /v1/systemone`. Both are adapters
over the `system-one` library in this directory, which owns every decision that depends on
the readout -- how many sequences to run, which tokens carry the answer, where the answer
sits -- so the two front ends cannot drift apart. `--system-one-request` takes the server's
request body verbatim, which is the way to check one against the other.

`temperature` is one number for the whole request, not one per question. A checkpoint ships
with its calibration already folded into its weights, so T = 1 is the normal case and is
skipped outright; the knob is there for a caller that has fitted a T on its own distribution,
which is a property of that deployment rather than of any one question. The raw `logits` are
in every answer, so anything more elaborate can be computed by the caller.

`system_one.readout` in the GGUF selects the readout:

| readout | the answer is | cost |
|---|---|---|
| `letter_slot` | the next-token distribution at the end of each question's segment | one pass |
| `masked_slot` | the distribution at a mask token inside it (bidirectional models) | one pass |
| `rank_head` | the model's classification head scoring one sequence per option | K passes |
