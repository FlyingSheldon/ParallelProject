import React from "react";
import {
  Typography,
  Container,
  Theme,
  Stack,
  Box,
} from "@mui/material";
import { makeStyles, createStyles } from "@mui/styles";
import { ImageControl } from "./ImageControl";

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
        <ImageControl />
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
