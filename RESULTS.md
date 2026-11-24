# Track B experiment results

The same agent used Gemini 3.1 Flash-Lite to generate an initial PyTorch experiment for each dataset. It then supplied the initial script and measured validation metrics to Gemini, which proposed a revision and explained its reasoning in the revised script. Each experiment trained on the fixed training split for two epochs. The held-out test splits were evaluated after both revisions were fixed.

| Dataset and metric | Initial validation | Revised validation | Revised held-out test |
|---|---:|---:|---:|
| Leaf Classification accuracy | 40.40% | 28.28% | 34.34% (34 of 99 images) |
| Document Denoising pixel RMSE | 0.2510 | 0.1747 | 0.1785 (14 pages) |

## Leaf Classification

The initial two-layer CNN reached 40.40% validation accuracy using raw leaf images. Gemini added a third convolutional layer and batch normalization to improve feature extraction and training stability. The revision also changed the pooling and classifier input. Under the same two-epoch budget, validation accuracy fell by 12.12 percentage points. On the held-out split, the revised model classified 34 of 99 images correctly. These results compare the training setups as a whole, since several architectural changes were made in one revision.

## Document Denoising

The initial three-layer CNN had a validation RMSE of 0.2510. Gemini doubled the internal channel count from 16 to 32 and added a residual connection so the network could learn a correction to the noisy input. Validation RMSE fell by approximately 30.4%. The revised model achieved 0.1785 RMSE on held-out pages. RMSE was computed across all pixels on full pages using intensities scaled to [0, 1]. The experiment measures the combined effect of the wider model and the residual connection.

## Execution and reproducibility

PyTorch intermittently crashed during backpropagation. The saved Gemini-generated scripts completed on replay without changing the model code or requesting a new revision. The submitted run directories contain the prompts, generated scripts, completed training logs, validation metrics, and checkpoints. The held-out evaluator reloads the revised checkpoints and runs inference without training. Its raw output is in `test_metrics.json`.
