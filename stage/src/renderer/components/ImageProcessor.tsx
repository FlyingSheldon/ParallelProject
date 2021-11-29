import React, { useEffect } from "react";
import {
    selectSharpness,
    selectBrightness,
    selectCurrentFile,
    selectOriginalFile,
    resetCurrentFile,
    fetchProcessedImage
} from "../store";
import { useAppSelector, useAppDispatch } from "../app/hooks";
import { DEFAULT_BRIGHTNESS, DEFAULT_SHARPNESS } from "../app/constants";

const ImageProcessor: React.FC = () => {
    const brightness = useAppSelector(selectBrightness)
    const sharpness = useAppSelector(selectSharpness)
    const originalFile = useAppSelector(selectOriginalFile)
    const currentFile = useAppSelector(selectCurrentFile)
    const dispatch = useAppDispatch()

    useEffect(() => {
        if (!originalFile) return
        if (brightness === DEFAULT_BRIGHTNESS
            && sharpness === DEFAULT_SHARPNESS
            && originalFile === currentFile) return


        if (brightness === DEFAULT_BRIGHTNESS
            && sharpness === DEFAULT_SHARPNESS
            && originalFile !== currentFile) {
            dispatch(resetCurrentFile())
        } else {
            dispatch(fetchProcessedImage({ filePath: originalFile.path, metrics: { brightness, sharpness } }))
        }
    }, [brightness, sharpness, originalFile, currentFile, dispatch])

    return <></>
}

export default ImageProcessor;