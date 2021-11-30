import { createSlice, PayloadAction, createAsyncThunk } from "@reduxjs/toolkit"
import { ImageMetrics } from "../../types";
import { AppState } from "./store"

export interface ImageFile {
    path: string,
    data: string
}

export interface FileState {
    originalFilePath?: string,
    originalFileData?: string,
    currentFilePath?: string,
    currentFileData?: string,
    currentFilePending: boolean,
}

const initialState: FileState = {
    currentFilePending: false
}

export const fetchOriginalFileData = createAsyncThunk(
    "file/fetchFileData",
    async () => {
        const res = await window.electronAPI.openImage();
        return res;
    }
)

export const fetchProcessedImage = createAsyncThunk(
    "file/fetchProcessedImage",
    async ({ filePath, metrics }: { filePath: string, metrics: ImageMetrics }) => {
        const res = await window.electronAPI.processImage(filePath, metrics);
        return res;
    }
)

export const fileSlice = createSlice({
    name: "file",
    initialState,
    reducers: {
        setOriginalFile: (state, action: PayloadAction<ImageFile>) => {
            state.originalFilePath = action.payload.path
            state.originalFileData = action.payload.data
            state.currentFilePath = action.payload.path
            state.currentFileData = action.payload.data
        },
        resetCurrentFile: (state) => {
            state.currentFileData = state.originalFileData
            state.currentFilePath = state.originalFilePath
        }
    },
    extraReducers: (builder) => {
        builder
            .addCase(fetchOriginalFileData.fulfilled, (state, action: PayloadAction<ImageFile | undefined>) => {
                if (action.payload) {
                    state.originalFileData = action.payload.data
                    state.originalFilePath = action.payload.path
                    state.currentFileData = action.payload.data
                    state.currentFilePath = action.payload.path
                }
            })
            .addCase(fetchProcessedImage.fulfilled, (state, action: PayloadAction<ImageFile | undefined>) => {
                if (action.payload) {
                    state.currentFileData = action.payload.data
                    state.currentFilePath = action.payload.path
                }
                state.currentFilePending = false
            })
            .addCase(fetchProcessedImage.pending, (state) => {
                state.currentFilePending = true
            })
    }
})

export const resetCurrentFile = fileSlice.actions.resetCurrentFile

export const selectFile = (state: AppState): FileState => state.file

export const selectOriginalFile = (state: AppState): ImageFile | undefined => {
    if (state.file.originalFileData) {
        return { path: state.file.originalFilePath, data: state.file.originalFileData }
    } else {
        return undefined
    }
}

export const selectCurrentFile = (state: AppState): ImageFile | undefined => {
    if (state.file.currentFileData) {
        return { path: state.file.currentFilePath, data: state.file.currentFileData }
    } else {
        return undefined
    }
}

export const selectCurrentFilePending = (state: AppState): boolean => {
    return state.file.currentFilePending
}

export default fileSlice.reducer