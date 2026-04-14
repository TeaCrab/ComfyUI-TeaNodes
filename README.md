# ComfyUI-TeaNodes

Adds a few new nodes:

### Image Equalization CLAHE (This node can't be any more Kornia)
*  When images have their histogram smoothly distributed, I'd say it gives ControlNet preprocessor an easier time.
*  Works great when image BG is removed.

### Image Size Approximation
*  Works based on pixel count that retains Image ratio.
*  This algorithm is really dumbly written, but it works.
*  Great for stabilizing the speed of image-to-image generation.

### Image Resize
*  Takes size tuple from Size Approximation Node.
*  Seriously, image size always has 2 numbers, why can't it fit through a single wire?
*  It also defaults to a node socket instead of having to convert it into input from a widget.

### Image Scale
*  Simply multiplies image size by a factor.
*  20240425: Now scales smoothly when multiplied by non-power-of-2 factors - use LANCZOS option now
*  Have an easier time saving in-process scrap images at half or quarter resolution.

### Crop To
*  Crop an image to the same size of the reference image
*  Hate some of those nodes that complains about bad tensor dimensions for whatever reasons? Fear no longer.

### KorniaGamma
*  This works like an easy/complex/idk brightness/contrast/level adjustment node
*  Image processing power must be utilized for better results.

### Random & Non-Random Color Fill
*  Fill based on reference image size
*  Randomness defined by original color and a variance in hue, saturation and value
*  Hopefully the color input field works for most color string formats.

### Random Lora & Model
<<<<<<< HEAD
*  Uses regex pattern to filter down files within `models/checkpoints` and `models/loras` and then randomly choose one to load.
*  `every` parameter allows the chosen model to ran certain number of times.
*  `pause` parameter allows the chosen model to run indefinitely.
*  `skip` parameter and changes in `pattern` parameter will trigger a random selection ASAP, `skip` doesn't stop loop counting, but changes in `pattern` will.
*  Models are pooled, which can lead to maxing out the RAM usage, Python seem to be able to handle this on its own without issue.
  * If a model has been used before in the same session, the node won't need to access the file from SSD/HDD again, it's already in the RAM.
*  `RESULT` outputs the working state of the node, use `Preview As Text` or `Show Text` node to understand time until next randomization, which model is being used currently, previously and how many times, or which ones hasn't been used yet from the found files.
*  If the regex pattern went wrong and there are no models found, a random model will be selected from all available under the respective `checkpoints` or `loras` folder.
=======
*  Uses regex pattern to filter down files within `models/checkpoints`, `models/loras` directory and then randomly choose one to load.
*  `every=#` parameter allows the chosen model to run only `#` number of times until randomly selecting another.
*  `pause=True` allows the current running model to run indefinitely.
*  `skip=True` or any changes in `pattern` will trigger a random selection ASAP, `skip` doesn't stop loop counting, but changes in `pattern` will.
*  Models are pooled, which can lead to maxing out the RAM usage, Python seem to be able to handle this on its own without issue.
    * If a model has been used before in the same session, the node won't need to access the file from SSD/HDD again, it's already in the RAM.
*  `RESULT` outputs the working state of the node, use `Preview As Text` or `Show Text` node to understand time until next randomization, which model is being used currently, previously and how many times, or which ones hasn't been used yet from the found files.
*  If the regex pattern went wrong or if there are no models found, a random model will be selected from all available under the respective `checkpoints` or `loras` folder.
>>>>>>> 0f38763 (Update README.md)
