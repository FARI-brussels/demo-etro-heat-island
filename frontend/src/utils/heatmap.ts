// utils/heatmap.ts
export interface CanvasData {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  imageData: ImageData;
}

/**
 * Creates a canvas and image data based on the source image dimensions.
 * @param sourceImage - The source HTMLImageElement.
 * @returns CanvasData object or null if creation fails.
 */
export function createCanvasImageData(sourceImage: HTMLImageElement): CanvasData | null {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    console.error('Failed to get canvas context');
    return null;
  }

  canvas.width = sourceImage.naturalWidth;
  canvas.height = sourceImage.naturalHeight;
  const imageData = ctx.createImageData(canvas.width, canvas.height);

  return { canvas, ctx, imageData };
}

/**
 * Normalizes a heat matrix to a 0-255 range for colormap application.
 * @param heatMatrix - 2D array of temperature values.
 * @param minVal - Minimum value in the matrix.
 * @param maxVal - Maximum value in the matrix.
 * @returns Normalized 2D array.
 */
export function normalizeMatrix(
  heatMatrix: number[][],
  minVal: number,
  maxVal: number
): number[][] {
  return heatMatrix.map(row =>
    row.map(value => {
      if (maxVal === minVal) return 0;
      return Math.floor(255 * (value - minVal) / (maxVal - minVal));
    })
  );
}

/**
 * Applies a plasma colormap to the normalized matrix and updates the canvas image data.
 * @param normalizedMatrix - Normalized 2D array (0-255).
 * @param imageData - Canvas ImageData object.
 * @param width - Canvas width.
 * @param height - Canvas height.
 */
export function applyPlasmaColormap(
  normalizedMatrix: number[][],
  imageData: ImageData,
  width: number,
  height: number
): void {
  const data = imageData.data;
  const scaleX = width / normalizedMatrix[0].length;
  const scaleY = height / normalizedMatrix.length;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const matrixX = Math.floor(x / scaleX);
      const matrixY = Math.floor(y / scaleY);
      const value = normalizedMatrix[matrixY]?.[matrixX] ?? 0;
      const idx = (y * width + x) * 4;

      let r, g, b;
      if (value < 85) {
        r = Math.floor(value * 3);
        g = 0;
        b = 255;
      } else if (value < 170) {
        r = 255;
        g = Math.floor((value - 85) * 3);
        b = Math.floor(255 - (value - 85) * 3);
      } else {
        r = 255;
        g = Math.floor(255 - (value - 170) * 3);
        b = 0;
      }

      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = 255;
    }
  }
}

/**
 * Rotates and flips the canvas to create the final heatmap image.
 * @param sourceCanvas - The source canvas with image data.
 * @returns Rotated canvas or null if creation fails.
 */
export function rotateAndFlipCanvas(sourceCanvas: HTMLCanvasElement): HTMLCanvasElement | null {
  const rotatedCanvas = document.createElement('canvas');
  const rotatedCtx = rotatedCanvas.getContext('2d');
  if (!rotatedCtx) {
    console.error('Failed to get rotated canvas context');
    return null;
  }

  rotatedCanvas.width = sourceCanvas.height;
  rotatedCanvas.height = sourceCanvas.width;

  rotatedCtx.save();
  rotatedCtx.translate(rotatedCanvas.width / 2, rotatedCanvas.height / 2);
  rotatedCtx.rotate(-Math.PI / 2);
  rotatedCtx.scale(-1, 1);
  rotatedCtx.drawImage(sourceCanvas, -sourceCanvas.width / 2, -sourceCanvas.height / 2);
  rotatedCtx.restore();

  return rotatedCanvas;
}

/**
 * Generates a heatmap image from a heat matrix.
 * @param heatMatrix - 2D array of temperature values.
 * @param minVal - Minimum value in the matrix.
 * @param maxVal - Maximum value in the matrix.
 * @param sourceImage - The source HTMLImageElement.
 * @returns Data URL of the heatmap image or null if generation fails.
 */
export function generateHeatmap(
  heatMatrix: number[][],
  minVal: number,
  maxVal: number,
  sourceImage: HTMLImageElement
): string | null {
  if (!sourceImage) {
    console.error('Source image not provided');
    return null;
  }

  const canvasData = createCanvasImageData(sourceImage);
  if (!canvasData) return null;

  const { canvas, ctx, imageData } = canvasData;
  const normalizedMatrix = normalizeMatrix(heatMatrix, minVal, maxVal);
  applyPlasmaColormap(normalizedMatrix, imageData, canvas.width, canvas.height);
  ctx.putImageData(imageData, 0, 0);

  const rotatedCanvas = rotateAndFlipCanvas(canvas);
  if (!rotatedCanvas) return null;

  return rotatedCanvas.toDataURL('image/png');
}