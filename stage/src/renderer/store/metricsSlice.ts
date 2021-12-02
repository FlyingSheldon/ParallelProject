import { createSlice, PayloadAction } from "@reduxjs/toolkit"
import { AppState } from "./store"
import { DEFAULT_BRIGHTNESS, DEFAULT_SHARPNESS } from "../app/constants"
import { fetchOriginalFileData, fetchProcessedImage } from "./fileSlice"
import { ImageMetrics } from "../../types"

export interface MetricsState {
    brightness: number;
    sharpness: number;
    currBrightness: number;
    currSharpness: number;
}

const initialState: MetricsState = {
    brightness: DEFAULT_BRIGHTNESS,
    sharpness: DEFAULT_SHARPNESS,
    currBrightness: DEFAULT_BRIGHTNESS,
    currSharpness: DEFAULT_SHARPNESS
}

export const metricsSlice = createSlice(
    {
        name: "metrics",
        initialState,
        reducers: {
            setBrightness: (state, action: PayloadAction<number>) => {
                state.brightness = action.payload
            },
            setSharpness: (state, action: PayloadAction<number>) => {
                state.sharpness = action.payload
            }
        },
        extraReducers: (builder) => {
            builder
                .addCase(fetchOriginalFileData.fulfilled, (state) => {
                    state.brightness = DEFAULT_BRIGHTNESS
                    state.sharpness = DEFAULT_SHARPNESS
                })
                .addCase(fetchProcessedImage.pending, (state) => {
                    state.currBrightness = state.brightness
                    state.currSharpness = state.sharpness
                })
        }
    }
)

export const { setBrightness, setSharpness } = metricsSlice.actions;

export const selectBrightness = (state: AppState): number => state.metrics.brightness

export const selectSharpness = (state: AppState): number => state.metrics.sharpness

export const selectMetrics = (state: AppState): ImageMetrics => {
    return {
        brightness: state.metrics.brightness,
        sharpness: state.metrics.sharpness
    }
}

export const selectCurrentMetrics = (state: AppState): ImageMetrics => {
    return {
        brightness: state.metrics.currBrightness,
        sharpness: state.metrics.currSharpness
    }
}


export default metricsSlice.reducer

