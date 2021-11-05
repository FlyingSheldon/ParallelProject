import React from "react";
import { Box, Grid, Stack, Button, Container } from "@mui/material";
import { makeStyles } from "@mui/styles";
import { selectOriginalFile, fetchOriginalFileData } from "../store";
import { useSelector, useDispatch } from "react-redux";

const useStyles = makeStyles({
  imgContainer: {
    height: "100%",
  },
  img: {
    minWidth: "60%",
    maxWidth: "80%",
  },
});

const Stage: React.FC = () => {
  const styles = useStyles();
  const dispatch = useDispatch();
  const originalFile = useSelector(selectOriginalFile);

  const handleOpen = () => {
    dispatch(fetchOriginalFileData());
  };

  const handleSave = async () => {
    if (originalFile) {
      await window.electronAPI.saveImage(originalFile.path);
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
          {originalFile && (
            <img src={originalFile.data} className={styles.img} />
          )}
        </Container>
      </Stack>
    </>
  );
};

export default Stage;
