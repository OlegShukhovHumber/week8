# PyTorch or Keras for the weekly bike demand job

Both versions are the same model. With copied weights their predictions on the 1,200 validation rows differ by at most 4.3e-08. Both were trained with identical rows and settings on seeds 2026, 7 and 42 (`reports/comparison.csv`).

## What I observed

1. **Accuracy is a tie.** Mean test MAE is 2.42 rentals for PyTorch and 2.40 for Keras. Keras was better on two seeds and worse on one, and the gap (0.02) is smaller than the spread between seeds (up to 0.08 for Keras).
2. **PyTorch was twice as fast.** It took 4.6–5.3 s per run and Keras 9.9–10.0 s, for 150 epochs on a laptop CPU. On a model this small that is mostly Keras's fixed cost per epoch, so it should not decide alone.
3. **Keras needs much less training code.** In PyTorch, `run_epoch` and `fit` are 43 lines I had to get right myself. In Keras it is one `compile` and one `fit` call, and the best weights come back through `EarlyStopping(restore_best_weights=True)`.
4. **Saved models differ.** A `.keras` checkpoint is about 100 KB and holds the architecture, so it loads without the code. A PyTorch `state_dict` is 28 KB but only holds the numbers, so loading it needs `model.py`.
5. **Debugging differs.** I can step through the PyTorch loop line by line. Every Keras `fit` printed a TensorFlow `NodeDef ... use_unbounded_threadpool` error from inside the library. It was harmless, but hard to trace.

## Recommendation: PyTorch

- A small model retrained weekly needs no Keras convenience: our loop is already written and tested (16 tests from Lab 07).
- The team may move to pretrained Hugging Face models later this year. Most are published for PyTorch first, so both projects would share one framework.
- With two people, one shared framework means either can fix the job.

## What we give up

- The Keras teammate has to learn PyTorch.
- We keep a hand-written loop that can contain silent mistakes (a forgotten `eval()`) that Keras callbacks avoid.
- A checkpoint cannot be loaded without our model code.

## What would change it

If the model had to run on a phone or in a browser, I would recommend Keras, because TensorFlow has long-established tools for those targets. I would also switch if the PyTorch teammate left, leaving only the Keras user to maintain the job.
