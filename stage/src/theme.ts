import { createTheme } from '@mui/material/styles';
// declare module '@mui/material/styles' {
//     interface Theme {
//         status: {
//             danger: string;
//         };
//         palette: {
//             mode: string;
//         };
//     }
//     // allow configuration using `createTheme`
//     interface ThemeOptions {
//         status?: {
//             danger?: string;
//         };
//     }
// }

const theme = createTheme({
    palette: {
        mode: "dark"
    }
});

export default theme;