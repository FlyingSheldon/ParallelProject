import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("electronAPI", {
  openImage: async () => {
    return await ipcRenderer.invoke("open-image");
  },
  saveImage: async (path: string) => {
    await ipcRenderer.invoke("save-image", path)
  }
});
