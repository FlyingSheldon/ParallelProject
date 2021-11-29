import React from "react";
import { selectSharpness, selectBrightness, selectCurrentFile, selectOriginalFile } from "../store";
import { useAppSelector } from "../app/hooks";

const ImageProcessor: React.FC = () => {
    const brightness = useAppSelector(selectBrightness)
    const sharpness = useAppSelector(selectSharpness)
    const originalFile = useAppSelector(selectOriginalFile)
    const currentFile = useAppSelector(selectCurrentFile)

    console.log(`Brightness: ${brightness}`)
    console.log(`Sharpness: ${sharpness}`)
    console.log(`Original File: ${originalFile?.path}`)
    console.log(`Current File: ${currentFile?.path}`)
    return <></>
}

export default ImageProcessor;