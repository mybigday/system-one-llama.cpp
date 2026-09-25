# System One

A System One model does not write text. Give it a **state** and a set of **typed questions**,
and one forward pass returns a calibrated distribution over each question's declared options.
Nothing is sampled and no token is generated, so the answer cannot be malformed by
construction -- there is no reply to parse and no schema to validate.

This directory holds the `system-one` library, which builds that prompt and reads those
answers, and `llama-system-one`, a CLI over it.

## Three question types

| type | the answer | read it as |
|---|---|---|
| `noul` | P(true) for an assertion | a threshold: act at > 0.8, ask again below 0.4 |
| `choice` | a distribution over declared options | routing, classification, intent |
| `score` | a distribution over ordered levels, plus its index-weighted mean | severity, satisfaction, priority |

Every answer also carries `confidence = 1 − H(p)/ln K` and its raw logits, because calibration
is the product: a consumer that only takes the argmax could have used anything.

## llama-system-one

```bash
llama-system-one -m model.gguf \
    --state "Support ticket: I was charged twice for the same subscription. Please refund the duplicate." \
    --noul   "refund:The customer is asking for a refund." \
    --choice "queue:Which team should handle this ticket?:billing=payments and refunds,technical=a product fault,account=login or profile settings" \
    --score  "urgency:How urgent is this?:routine,soon,urgent,critical"
```

```
refund  (noul, confidence 0.578)
  P(true) = 0.9141

queue  (choice, confidence 0.910)
  -> billing
     billing                  0.9824
     technical                0.0129
     account                  0.0046

urgency  (score, confidence 0.369)
  score = 1.8675 on 0..3  (0 routine .. 3 critical)
     routine                  0.0802
     soon                     0.0803
     urgent                   0.7312
     critical                 0.1083

138 prompt tokens, 1 sequence(s), 0 tokens generated
```

Three decisions out of one prompt: whether to open a refund case, which queue to route it to,
and how to prioritise it -- each a number a caller can threshold, and no text to parse.

**The answers are the checkpoint's, not this tool's.** That run is a Qwen3-based checkpoint;
the same questions put to a 270M model trained on zh-TW data return P(true) 0.28 for the
refund assertion, which is wrong. A System One model is only calibrated on the distribution it
was trained for, so check a checkpoint against your own questions, in the language you will
actually serve, before trusting its numbers.

| option | |
|---|---|
| `--state TEXT` / `--state-file FNAME` | the context the questions are asked about |
| `--noul KEY:INSTRUCTIONS` | a true/false question; repeatable |
| `--choice KEY:INSTRUCTIONS:opt[=desc][,...]` | pick one declared option; repeatable |
| `--score KEY:INSTRUCTIONS:level[,...]` | ordered levels, and the mean of their indices |
| `--system-one-request FNAME` | the whole request as JSON, instead of the flags |
| `--system-one-template-file FNAME` | override the checkpoint's System One template |
| `--labels A,B,C` | override the labels an answer is read at (default A-Za-z) |
| `--system-one-temperature T` | rescale every answer's logits by T (default 1, skipped) |
| `--json` | print the answers as JSON |

`--json` is the same shape either front end returns:

```jsonc
{"model": "...",
 "answers": {
   "refund":  {"noul": 0.9141, "confidence": 0.5775, "logits": [9.7280, 12.0933]},
   "queue":   {"choice": "billing",
               "probabilities": {"billing": 0.9824, "technical": 0.0129, "account": 0.0046},
               "confidence": 0.9104, "logits": [13.4224, 9.0917, 8.0633]},
   "urgency": {"score": 1.8675, "legend": ["routine", "soon", "urgent", "critical"],
               "probabilities": [0.0802, 0.0803, 0.7312, 0.1083],
               "confidence": 0.3691, "logits": [9.8189, 9.8197, 12.0286, 10.1188]}},
 "usage": {"input_tokens": 138, "output_tokens": 0, "sequences": 1}}
```

`output_tokens: 0` is the point of the model class. `--system-one-request` takes that request
side back, which is how a prompt is kept identical between this CLI and a server.

The instructions in `--noul` / `--choice` / `--score` may not contain a colon; that is what the
JSON request is for, along with anything else more elaborate than a one-off from the shell.

## llama-system-one-diffusion

DiffusionGemma does not fit the readouts above, and it has its own command rather than a flag.
It is an encoder-decoder: a causal encoder over the prompt, then a **canvas** of
`diffusion.canvas_length` positions denoised bidirectionally against the encoder's K,V. The
answer is read from the canvas, not from inside the prompt.

A plain `llama_decode` -- which is all the library and the server ever issue -- lands in the
arch's unified branch, where the region split is positional:

```
P = n_tokens - canvas_length
```

so the boundary falls wherever the prompt happens to end. On a 380-token request with a
256-token canvas that is token 124, in the middle of the state: part of the state gets denoised
as if it were an answer, and the first question's mask sits on the causal-prompt side. The
numbers that come out are a faithful reproduction of a configuration the model was never
trained in.

This command drives the model's own path instead. The template writes both regions, separated
by a segment that is exactly the canvas marker (`<|canvas|>` by default): the state and the
question *text* go in front of it, and after it one labelled, masked answer slot per question
(`q1: <mask>`), which is how the model is used in the wild.

```
llama-system-one-diffusion model.gguf golden.json --template-file tmpl.jinja \
    [--mode prefill|unified|both] [--pad] [--dry-run] [--control] [--dump-ids PFX] \
    [--limit N] [--threads N] [--dump out.json]
```

`--dry-run` loads the vocabulary only and prints how the template split, so the layout can be
checked without the weights. `--mode both` runs the cached path and the single-batch path and
reports the largest disagreement at the answer slots, judged over each question's own option
letters rather than the whole vocabulary; `--control` repeats each path so you can see it
reproduces itself before reading anything into the difference. `--dump-ids` writes the ids in the
raw int32 format `llama-diffusion-gemma-eval` takes, so the same layout can go through the arch's
own harness.

**The canvas keeps its rendered length by default, and padding is a flag.** `unified` splits on
the GGUF `canvas_length` so it has no choice, but DECODE takes the canvas length from the batch --
and padding is not neutral. Measured against HF on the same prompt, padding a 30-token canvas out
to 256 with mask tokens moves the model's own answer probabilities by up to **.386** (argmax
unchanged). Pad only to compare the two paths, which have to see the same canvas.

Two things to know before reading a number off this model. Its logits are **extremely
precision-sensitive**: rounding the weights bf16 -> f16 with identical arithmetic moves them by
2.844 on average against a +-30 softcap, so an f16 GGUF of it cannot be validated at the logit
level at all. And its two paths **disagree with each other** on identical weights (mean 1.9 at
bf16), which the arch's own source says cannot happen. Where the readout actually reads -- answer
slots, each question's own options, bf16, unpadded -- llama.cpp and HF agree to |dp| .104 with the
same argmax on all five questions.

It is a separate command because the phase it drives (`llama_diffusion_set_phase`) belongs to
one arch that is not upstream, and because that phase is set on the *model*, which a server
shares between slots.

## The prompt format belongs to the checkpoint

It is a Jinja template stored in the GGUF as the named chat template `system_one`, rendered
with llama.cpp's own Jinja engine. There is no built-in default, so a model without one is an
error. `--system-one-template-file` overrides it, and usually needs `--labels` with it: the labels say *where* an answer is read, so a format that writes
`(1)(2)(3)` has to say so or the reply is a distribution over `A`, `B`, `C` that nothing wrote.

This is **not** the chat template. A System One template is rendered with a state, questions
and option labels, never with a message list.

Segment boundaries matter as much as the text. The reference implementations encode the state
and each question block separately and concatenate the ids, which differs from a single-pass
tokenization at the seams; the template marks those seams with `system_one.segment_separator`
(U+001E) and this side tokenizes each piece on its own. On the 255-state reference set that
costs exactly one token per prompt against a single-pass encode -- and matching it is what
makes the numbers comparable to the reference implementation at all.

## Readouts

Where the answer sits is **derived from what the checkpoint is**, and there is no metadata key
for it: a model carrying a classification head is read through it, a bidirectional one at a
mask token, a causal one at the next token. So a checkpoint read the wrong way is a checkpoint
describing itself wrongly -- almost always `{arch}.attention.causal`, which the graph builds
its attention mask from too.

| readout | the answer is | cost |
|---|---|---|
| `letter_slot` | the next-token distribution at the end of each question's segment | one pass |
| `masked_slot` | the distribution at a mask token inside it (bidirectional models) | one pass |
| `scored_slot` | one number per position from the model's own head, read at the positions it marks | one pass **per question** |
| `rank_head` | the model's classification head scoring one sequence per option | K passes |

Only `letter_slot` can reuse a prefix across calls: bidirectional attention makes every
position depend on what follows it, so changing the tail of a state changes its head.

`scored_slot` needs a server started with `--embeddings --pooling none`, because its answer
arrives as embeddings rather than logits.

## Asking several questions at once

This section is about `letter_slot` and `masked_slot`, the two readouts that put every question
in one prompt. `scored_slot` renders one sequence per question and `rank_head` one per option, so
for those the question below does not arise -- and neither does the speed argument.

All of a request's questions are slots in one prompt, so they are answered by one forward pass:
five questions about a 214-token ticket cost one decode, and asking them separately re-encodes
the state each time. Later questions can see the text of earlier ones but never their answers --
each block ends at the answer slot with nothing written into it.

**This depends on the checkpoint having been trained for it.** A model that was not will answer
the first slot and then stop answering: having seen one or more `Answer k: (` lines that were
never followed by a letter, it concludes that answers stay empty in this document and continues
to the next question instead. The readout still reads the label logits at that position, and
they are now the logits of a model that has decided not to answer. Measured over 120 states,
5 questions each:

| | all in one prompt | one question per request |
|---|---|---|
| a checkpoint trained for this format | .973 | .965 |
| stock Qwen3.5-0.8B, template supplied by the request (`letter_slot`) | .544 | .699 |
| stock gemma-4 E2B, template supplied by the request (`letter_slot`) | .541 | .765 |
| stock Qwen3-0.6B-mdlm (`masked_slot`, bidirectional) | .494 | .525 |

So: point this at a stock model and ask **one question per request**; the one-pass form is for a
checkpoint trained on it, where it is both the faster mode (629 ms against 1049 ms for five
questions) and the more accurate one.

The last row is the same effect at a fifth the size, and it says where the effect comes from: the
15-to-22-point version is a *causal* model inducing from its own unanswered slots, which needs a
privileged left context. A bidirectional model cannot induce that, and keeps only the residual
from sharing a prompt at all. `scored_slot` and `rank_head` have neither, by construction -- they
never put two questions in one sequence -- and reproduce their one-pass accuracy exactly.

Distance is not the variable, for either kind of model -- moving a slot from 35 to 716 tokens
away from the state changes a trained checkpoint's answers by a mean |ΔP| of .039 to .043, which
is no change at all. What matters is whether another question sits in front of it.

## Temperature

`temperature` rescales every answer's logits before the probabilities are taken; it is one
number for the whole request. A checkpoint ships with its calibration already folded into its
weights, so T = 1 is the normal case and is skipped outright -- the knob is for a caller that
has fitted a T on its own distribution. The raw logits are in every answer, so anything more
elaborate is the caller's to compute.

It is not `--temp`, the sampling temperature, which means nothing here: nothing is sampled.
