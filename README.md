# ComfyUI-TeaNodes

Adds a few new nodes:

## Image Equalization CLAHE style.
*  When images have their histogram smoothly distributed, I'd say it gives ControlNet preprocessor an easier time.
*  Works great when image BG is removed.

## Image Size Approximation based on pixel count that retains Image ratio.
*  Great for stabilizing the speed of image-to-image generation.

## Image Resize Node that takes size tuple from Size Approximation Node.
*  Seriously, image size always has 2 numbers, why can't it fit through a single wire?
*  It also defaults to a node socket instead of having to convert it into input from a widget.

## Image Scale Node simply multiplies image size by a factor.
*  Have an easier time saving in-process scrap images at half or quarter resolution.

## Crop To - Crop an image to the same size of the reference image
*  Hate some of those nodes that complains about bad tensor dimensions for whatever reasons? Fear no longer.

## KorniaGamma - This works like an easy/complex/idk brightness/contrast/level adjustment node
*  Image processing power must be utilized for better results.

Don't know how licensing works so code relying on other repos has been ignored for now.
