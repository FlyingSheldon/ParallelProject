import { execSync } from "child_process";
import { dialog, BrowserWindow, IpcMainInvokeEvent } from "electron";
import { promises as fs } from 'fs';
import path from 'path';
import { ImageMetrics } from "../types";
import { buildPPCommand } from "./proc";

type OpenImageReturn = { path: string, data: string } | undefined

const getDataType = (p: string): string => {
    let dataType: string;
    const ext = path.extname(p).toLowerCase();
    if (ext === ".jpg" || ext === ".jpeg") {
        dataType = "jpg";
    } else if (ext === ".gif") {
        dataType = "gif";
    } else {
        dataType = "png";
    }
    return dataType
}

export const handleOpenImage = async (): Promise<OpenImageReturn> => {
    const res = await dialog.showOpenDialog(BrowserWindow.getFocusedWindow(), {
        properties: ['openFile'],
        filters: [
            { name: 'Images', extensions: ['jpg', 'jpeg'] }
        ]
    });

    if (res.canceled) {
        return undefined
    }

    const data = await fs.readFile(res.filePaths[0], { encoding: 'base64' });

    const dataType: string = getDataType(res.filePaths[0])

    return {
        path: res.filePaths[0],
        data: `data:image/${dataType};base64,${data}`,
    }
}

export const handleSaveImage = async (event: IpcMainInvokeEvent, srcPath: string): Promise<void> => {
    const defaultName = path.basename(srcPath)
    const res = await dialog.showSaveDialog(BrowserWindow.getFocusedWindow(), { defaultPath: defaultName });
    if (res.canceled) {
        return
    }

    const dstPath = res.filePath;

    await fs.copyFile(srcPath, dstPath)
}

export const handleProcessImage = async (event: IpcMainInvokeEvent, srcPath: string, metrics: ImageMetrics): Promise<OpenImageReturn> => {
    const fileName = path.basename(srcPath);
    const newPath = path.join(global.__TEMP_DIR_PATH__, fileName);
    const cmd = buildPPCommand(srcPath, newPath, metrics)

    execSync(cmd)

    const data = await fs.readFile(newPath, { encoding: 'base64' });
    const dataType: string = getDataType(srcPath)

    return {
        path: newPath,
        data: `data:image/${dataType};base64,${data}`,
    }
}