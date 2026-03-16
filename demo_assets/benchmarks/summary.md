# EdgeGen Benchmark Summary

This is a prototype-level quality evaluation based on handcrafted feature distance.

- Best pruned model by proxy quality: `best_pruned`
- Best pruned quality score: `0.1914`
- Best pruned feature distance: `4.2241`

| model_name   | artifact_path                                                                   | load_mode        |   model_size_mb |   avg_latency_ms |   throughput_samples_per_sec |   peak_memory_mb |   feature_distance |   quality_score | visual_quality_note                                                                         |
|:-------------|:--------------------------------------------------------------------------------|:-----------------|----------------:|-----------------:|-----------------------------:|-----------------:|-------------------:|----------------:|:--------------------------------------------------------------------------------------------|
| best_pruned  | /Users/soham/Downloads/prototype2/edgegen/outputs/checkpoints/best_pruned.pt    | dense            |        2.16775  |          1.50893 |                     10603.5  |         0        |            4.22405 |        0.191422 | Selected compressed quality checkpoint; best overall pruning trade-off for presentation.    |
| pruned_20    | /Users/soham/Downloads/prototype2/edgegen/outputs/checkpoints/pruned_20.pt      | dense            |        2.16775  |          1.55519 |                     10288.1  |         0.078125 |            4.71499 |        0.174978 | Light compression; typically closest visual match to baseline.                              |
| quantized    | /Users/soham/Downloads/prototype2/edgegen/outputs/checkpoints/quantized_int8.pt | weight_only_int8 |        0.547818 |          1.56616 |                     10216.1  |         0        |            4.74226 |        0.174148 | Most compact artifact; inspect for mild contrast or texture loss after dequantized loading. |
| pruned_40    | /Users/soham/Downloads/prototype2/edgegen/outputs/checkpoints/pruned_40.pt      | dense            |        2.16775  |          1.56862 |                     10200.1  |         0        |            4.43335 |        0.184049 | Moderate compression; balanced output with some softening.                                  |
| baseline     | /Users/soham/Downloads/prototype2/edgegen/outputs/checkpoints/best.pt           | dense            |        7.74683  |          1.5721  |                     10177.5  |         0.203125 |            4.82168 |        0.171772 | Baseline quality model; use this as the primary visual reference.                           |
| pruned_60    | /Users/soham/Downloads/prototype2/edgegen/outputs/checkpoints/pruned_60.pt      | dense            |        2.16775  |          1.79806 |                      8898.49 |         0        |            4.24444 |        0.190678 | Aggressive compression with noticeable detail risk but strong edge efficiency.              |

## Interpretation

- Lower latency and higher throughput indicate better CPU deployment efficiency.
- Lower feature distance and higher quality score indicate better prototype-level sample fidelity.
- Quantized runtime uses dequantized float32 execution when ConvTranspose kernels limit native int8 inference.
