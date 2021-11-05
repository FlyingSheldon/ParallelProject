import React, { useState } from "react";
import { Typography, Button } from "@mui/material";

const Stage: React.FC = () => {
  const [imgPath, setImgPath] = useState("");
  const [imgData, setImgData] = useState("");

  const handleOpen = async () => {
    const res = await window.electronAPI.openImage();
    setImgPath(res.path);
    setImgData(res.data);
  };

  return (
    <>
      <Typography variant="h3">This is the stage</Typography>
      <Button onClick={handleOpen}>Open</Button>
      {imgPath !== "" && <img src={imgData} />}
    </>
  );
};

export default Stage;
