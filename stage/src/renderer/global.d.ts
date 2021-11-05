export interface IElectronAPI {
    openImage: () => Promise<{ path: string, data: string }?>
    saveImage: (path: string) => Promise<void>
}

declare global {
    interface Window {
        electronAPI: IElectronAPI
    }
}