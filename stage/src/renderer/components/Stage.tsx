import React from "react";
import { Box, Grid, Stack, Button, Container, CircularProgress, Backdrop } from "@mui/material";
import { makeStyles } from "@mui/styles";
import { selectCurrentFile, fetchOriginalFileData, selectCurrentFilePending } from "../store";
import { useAppDispatch, useAppSelector } from "../app/hooks";

const useStyles = makeStyles({
  imgContainer: {
    height: "100%",
  },
  img: {
    minWidth: "50%",
    maxWidth: "80%",
  },
});

const Stage: React.FC = () => {
  const styles = useStyles();
  const dispatch = useAppDispatch();
  const currentFile = useAppSelector(selectCurrentFile);
  const isPending = useAppSelector(selectCurrentFilePending)

  const handleOpen = () => {
    dispatch(fetchOriginalFileData());
  };

  const handleSave = async () => {
    if (currentFile) {
      await window.electronAPI.saveImage(currentFile.path);
    }
  };

  return (
    <>
      <Stack sx={{ height: "100%" }}>
        <Grid container sx={{ p: 2 }}>
          <Button variant="contained" onClick={handleOpen}>
            Open
          </Button>
          <Box sx={{ m: 1 }} />
          <Button variant="contained" onClick={handleSave}>
            Save
          </Button>
        </Grid>
        <Container
          className={styles.imgContainer}
          sx={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          {currentFile && (
            <img src={currentFile.data} className={styles.img} />
          )}
          {isPending && <Backdrop
            sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
            open={isPending}
          >
            <CircularProgress color="inherit" />
          </Backdrop>}
        </Container>
      </Stack>
    </>
  );
};

export default Stage;
