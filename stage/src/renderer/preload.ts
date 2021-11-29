import { contextBridge, ipcRenderer } from "electron";
import { ImageMetrics } from "../types"

contextBridge.exposeInMainWorld("electronAPI", {
  openImage: async () => {
    return await ipcRenderer.invoke("open-image");
  },
  saveImage: async (path: string) => {
    await ipcRenderer.invoke("save-image", path)
  },
  processImage: async (path: string, metrics: ImageMetrics) => {
    await ipcRenderer.invoke("process-image", path, metrics)
  }
});
