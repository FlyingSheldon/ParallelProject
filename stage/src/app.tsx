import React from "react";
import ReactDom from "react-dom";
import theme from "./theme";
import { ThemeProvider, Grid, CssBaseline } from "@mui/material";
import { makeStyles } from "@mui/styles";
import Stage from "./components/Stage";
import Control from "./components/Control";
import { Provider } from "react-redux";
import store from "./store/store";

const useStyles = makeStyles({
  app: {
    height: "100%",
  },
});

const App = () => {
  const styles = useStyles();
  return (
    <>
      <Provider store={store}>
        <Grid container className={styles.app}>
          <Grid item xs={8}>
            <Stage />
          </Grid>
          <Grid item xs={4}>
            <Control />
          </Grid>
        </Grid>
      </Provider>
    </>
  );
};

const render = () => {
  ReactDom.render(
    <React.StrictMode>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <App />
      </ThemeProvider>
    </React.StrictMode>,
    document.body
  );
};

render();
