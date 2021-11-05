export interface IElectronAPI {
    openImage: () => Promise<{ path: string, data: string }>
}

declare global {
    interface Window {
        electronAPI, IElectronAPI
    }
}