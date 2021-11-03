import React, { useState } from "react";
import { Slider, Typography } from "@mui/material";
import { makeStyles } from "@mui/styles";

const useStyles = makeStyles({
  line: {
    display: "flex",
    flexDirection: "row",
    justifyContent: "space-between",
  },
});

export interface AdjustSliderProps {
  defaultValue?: number;
  label: string;
}

export const AdjustSlider: React.FC<AdjustSliderProps> = ({
  label,
  defaultValue = 50,
}) => {
  const styles = useStyles();
  const [value, setValue] = useState(defaultValue);

  const handleChange = (event: Event, newValue: number) => {
    setValue(newValue);
  };

  return (
    <>
      <div className={styles.line}>
        <Typography variant="h6">{label}</Typography>
        <Typography variant="h6" component="p">
          {value}
        </Typography>
      </div>
      <Slider value={value} onChange={handleChange} />
    </>
  );
};
