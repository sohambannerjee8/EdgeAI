# EdgeGen Professor Summary

This report condenses the CPU deployment comparison for presentation use.

| Model | Size (MB) | Latency (ms) | Throughput (samples/s) | Peak Memory (MB) | Visual Observation |
|:--|--:|--:|--:|--:|:--|
| best_pruned | 2.168 | 1.509 | 10603.52 | 0.000 | Selected compressed quality checkpoint; best overall pruning trade-off for presentation. |
| pruned_20 | 2.168 | 1.555 | 10288.12 | 0.078 | Light compression; typically closest visual match to baseline. |
| quantized | 0.548 | 1.566 | 10216.08 | 0.000 | Most compact artifact; inspect for mild contrast or texture loss after dequantized loading. |
| pruned_40 | 2.168 | 1.569 | 10200.07 | 0.000 | Moderate compression; balanced output with some softening. |
| baseline | 7.747 | 1.572 | 10177.47 | 0.203 | Baseline quality model; use this as the primary visual reference. |
| pruned_60 | 2.168 | 1.798 | 8898.49 | 0.000 | Aggressive compression with noticeable detail risk but strong edge efficiency. |

Prototype note: visual observations are qualitative and should be interpreted alongside saved sample grids.
