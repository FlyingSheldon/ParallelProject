import { createSlice, PayloadAction, createAsyncThunk } from "@reduxjs/toolkit"
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
}

const initialState: FileState = {}

export const fetchOriginalFileData = createAsyncThunk(
    "file/fetchFileData",
    async () => {
        const res = await window.electronAPI.openImage();
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
    }
})

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

export default fileSlice.reducer