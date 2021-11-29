import { createSlice, PayloadAction } from "@reduxjs/toolkit"
import { AppState } from "./store"
import { DEFAULT_BRIGHTNESS, DEFAULT_SHARPNESS } from "../app/constants"

export interface MetricsState {
    brightness: number;
    sharpness: number;
}

const initialState: MetricsState = {
    brightness: DEFAULT_BRIGHTNESS,
    sharpness: DEFAULT_SHARPNESS,
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
        }
    }
)

export const { setBrightness, setSharpness } = metricsSlice.actions;

export const selectBrightness = (state: AppState): number => state.metrics.brightness

export const selectSharpness = (state: AppState): number => state.metrics.sharpness


export default metricsSlice.reducer

