import React from "react";
import {
    Typography,
    Box,
    Divider,
} from "@mui/material";
import { AdjustSlider } from "./AdjustSlider";
import { setBrightness, setSharpness } from "../store/metricsSlice";
import { useAppDispatch } from "../app/hooks";

export const ImageControl: React.FC = () => {
    const dispatch = useAppDispatch()

    const brightnessHandler = (v: number) => {
        dispatch(setBrightness(v))
    }

    const sharpnessHandler = (v: number) => {
        dispatch(setSharpness(v))
    }

    return (
        <>
            <Box sx={{ m: 2 }} />
            <Typography variant="h5">Light</Typography>
            <Box sx={{ m: 1 }} />
            <AdjustSlider label="Brightness" defaultValue={0} min={-1} max={1} onChangeCommitted={brightnessHandler} />
            <Box sx={{ m: 0.5 }} />
            <AdjustSlider label="Highlight" />
            <Box sx={{ m: 0.5 }} />
            <AdjustSlider label="Shadow" />
            <Box sx={{ m: 2 }} />
            <Typography variant="h5">Sharpening</Typography>
            <Box sx={{ m: 1 }} />
            <AdjustSlider label="Sharpness" defaultValue={0} min={0} max={1} onChangeCommitted={sharpnessHandler} />
            <Box sx={{ m: 1 }} />
            <Divider />
            <Box sx={{ m: 1 }} />
        </>
    )
}