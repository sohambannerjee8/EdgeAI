# EdgeGen Model Comparison

All model variants below were generated from the same saved latent vectors for a fair visual comparison.

- shared latent file: `/Users/soham/Downloads/prototype2/edgegen/outputs/samples/comparisons/shared_latents.pt`
- combined comparison: `/Users/soham/Downloads/prototype2/edgegen/outputs/samples/comparisons/combined_comparison.png`

## baseline

- artifact: `/Users/soham/Downloads/prototype2/edgegen/outputs/checkpoints/best.pt`
- load mode: `dense`
- shared latents: `shared_latents.pt`
- image: `baseline_grid.png`
- note: Quality reference model with the fullest detail retained.

## pruned_20

- artifact: `/Users/soham/Downloads/prototype2/edgegen/outputs/checkpoints/pruned_20.pt`
- load mode: `dense`
- shared latents: `shared_latents.pt`
- image: `pruned_20_grid.png`
- note: Light pruning; visuals should stay close to baseline with mild softening.

## pruned_40

- artifact: `/Users/soham/Downloads/prototype2/edgegen/outputs/checkpoints/pruned_40.pt`
- load mode: `dense`
- shared latents: `shared_latents.pt`
- image: `pruned_40_grid.png`
- note: Moderate pruning; expect cleaner speed-size tradeoff with some loss of sharpness.

## pruned_60

- artifact: `/Users/soham/Downloads/prototype2/edgegen/outputs/checkpoints/pruned_60.pt`
- load mode: `dense`
- shared latents: `shared_latents.pt`
- image: `pruned_60_grid.png`
- note: Aggressive pruning; inspect for softer boundaries and reduced garment structure.

## quantized

- artifact: `/Users/soham/Downloads/prototype2/edgegen/outputs/checkpoints/quantized_int8.pt`
- load mode: `weight_only_int8`
- shared latents: `shared_latents.pt`
- image: `quantized_grid.png`
- note: Weight-only int8 artifact; inspect for contrast shifts after dequantized runtime loading.
