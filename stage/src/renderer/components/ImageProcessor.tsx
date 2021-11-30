import React, { useEffect } from "react";
import {
    selectCurrentFile,
    selectMetrics,
    selectCurrentMetrics,
    selectOriginalFile,
    resetCurrentFile,
    fetchProcessedImage
} from "../store";
import { useAppSelector, useAppDispatch } from "../app/hooks";
import { DEFAULT_BRIGHTNESS, DEFAULT_SHARPNESS } from "../app/constants";

const ImageProcessor: React.FC = () => {
    const metrics = useAppSelector(selectMetrics)
    const currMetrics = useAppSelector(selectCurrentMetrics)
    const originalFile = useAppSelector(selectOriginalFile)
    const currentFile = useAppSelector(selectCurrentFile)
    const dispatch = useAppDispatch()

    useEffect(() => {
        if (!originalFile) return

        if (metrics.brightness === currMetrics.brightness
            && metrics.sharpness === currMetrics.sharpness) return

        if (metrics.brightness === DEFAULT_BRIGHTNESS
            && metrics.sharpness === DEFAULT_SHARPNESS
            && originalFile !== currentFile) {
            dispatch(resetCurrentFile())
        } else {
            dispatch(fetchProcessedImage({ filePath: originalFile.path, metrics }))
        }
    }, [metrics, currMetrics, originalFile, currentFile, dispatch])

    return <></>
}

export default ImageProcessor;