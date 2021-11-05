import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("electronAPI", {
  openImage: async () => {
    return await ipcRenderer.invoke("open-image");
  },
});
