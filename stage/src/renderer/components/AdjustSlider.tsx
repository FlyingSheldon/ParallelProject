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
  min?: number;
  max?: number;
  label: string;
}

export const AdjustSlider: React.FC<AdjustSliderProps> = ({
  label,
  defaultValue = 50,
  min = 0,
  max = 100,
}) => {
  const styles = useStyles();
  const [value, setValue] = useState<number>(defaultValue);

  const step = (max - min) / 100
  const fmtValue = step < 1 ? value.toFixed(2) : value.toString();

  const handleChange = (event: Event, newValue: number) => {
    setValue(newValue);
  };

  return (
    <>
      <div className={styles.line}>
        <Typography variant="h6">{label}</Typography>
        <Typography variant="h6" component="p">
          {fmtValue}
        </Typography>
      </div>
      <Slider value={value} min={min} max={max} step={step} onChange={handleChange} />
    </>
  );
};
