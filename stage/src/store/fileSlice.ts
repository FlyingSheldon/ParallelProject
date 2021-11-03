import { createSlice, PayloadAction } from "@reduxjs/toolkit"
import { AppState } from "./store"

export interface FileState {
    originalFile?: File
}

const initialState: FileState = {}

export const fileSlice = createSlice({
    name: "file",
    initialState,
    reducers: {
        setOriginalFile: (state, action: PayloadAction<File>) => {
            state.originalFile = action.payload
        }
    }
})

export const selectFile = (state: AppState): FileState => state.file

export const selectOriginalFile = (state: AppState): File => state.file.originalFile

export default fileSlice.reducer