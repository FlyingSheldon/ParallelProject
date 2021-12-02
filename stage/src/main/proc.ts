import { ImageMetrics } from "../types"

export const buildPPCommand = (srcPath: string, dstPath: string, metrics: ImageMetrics): string => {
    const argv: string[] = [global.__PP_PATH__, srcPath, "-o", dstPath];
    argv.push("-brightness", (metrics.brightness + 1.0).toString());
    argv.push("-sharpness", metrics.sharpness.toString())

    return argv.join(' ');
}