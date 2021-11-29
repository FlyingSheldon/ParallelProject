import { ImageMetrics } from "../types";

export interface IElectronAPI {
    openImage: () => Promise<{ path: string, data: string }?>
    saveImage: (path: string) => Promise<void>
    processImage: (path: string, metrics: ImageMetrics) => Promise<{ path: string, data: string }?>
}

declare global {
    interface Window {
        electronAPI: IElectronAPI
    }
}