import React from "react";
import {
  Typography,
  Container,
  Theme,
  Stack,
  Box,
  Divider,
  stepButtonClasses,
} from "@mui/material";
import { makeStyles, createStyles } from "@mui/styles";
import { AdjustSlider } from "./AdjustSlider";

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    root: {
      backgroundColor: theme.palette.grey[900],
      padding: "2rem",
      height: "100%",
    },
    line: {
      display: "flex",
      flexDirection: "row",
      justifyContent: "space-between",
    },
  })
);

const Control: React.FC = () => {
  const styles = useStyles();
  return (
    <Container className={styles.root}>
      <Stack>
        <Typography variant="h4">Control</Typography>
        <Box sx={{ m: 2 }} />
        <Typography variant="h5">Light</Typography>
        <Box sx={{ m: 1 }} />
        <AdjustSlider label="Brightness" />
        <Box sx={{ m: 0.5 }} />
        <AdjustSlider label="Highlight" />
        <Box sx={{ m: 0.5 }} />
        <AdjustSlider label="Shadow" />
        <Box sx={{ m: 2 }} />
        <Typography variant="h5">Sharpening</Typography>
        <Box sx={{ m: 1 }} />
        <AdjustSlider label="Amount" />
        <Box sx={{ m: 0.5 }} />
        <AdjustSlider label="Radius" />
        <Box sx={{ m: 1 }} />
        <Divider />
        <Box sx={{ m: 1 }} />
        <Typography variant="h4">Stat</Typography>
        <Box sx={{ m: 2 }} />
        <div className={styles.line}>
          <Typography variant="h6">Execution Time</Typography>
          <Typography variant="h6">0s</Typography>
        </div>
      </Stack>
    </Container>
  );
};

export default Control;
