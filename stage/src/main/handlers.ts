import { dialog, BrowserWindow } from "electron";
import { promises as fs } from 'fs';
import path from 'path';

type OpenImageReturn = { path: string, data: string } | undefined

export const handleOpenImage = async (): Promise<OpenImageReturn> => {
    const res = await dialog.showOpenDialog(BrowserWindow.getFocusedWindow(), {
        properties: ['openFile'],
        filters: [
            { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'gif'] }
        ]
    });

    if (res.canceled) {
        return undefined
    }

    const data = await fs.readFile(res.filePaths[0], { encoding: 'base64' });

    let dataType: string;
    const ext = path.extname(res.filePaths[0]).toLowerCase();
    if (ext === ".jpg" || ext === ".jpeg") {
        dataType = "jpg";
    } else if (ext === ".gif") {
        dataType = "gif";
    } else {
        dataType = "png";
    }

    return {
        path: res.filePaths[0],
        data: `data:image/${dataType};base64,${data}`,
    }
}