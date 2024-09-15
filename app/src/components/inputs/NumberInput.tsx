import { FC, useEffect, useState } from "react"

export interface NumberInputProps {
  value: number
  min?: number
  max?: number
  step?: number
  id?: string
  onChange: (value: number) => void
  onEnter?: () => void
  style?: React.CSSProperties
  className?: string
}

export const NumberInput: FC<NumberInputProps> = ({
  value,
  min,
  max,
  step,
  id,
  onChange,
  onEnter,
  style,
  className,
}) => {
  const [rawValue, setRawValue] = useState(value)

  const onKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault()
      e.currentTarget.blur()
      onEnter?.()
    }
  }

  const updateRawValue = (newValue: number) => {
    setRawValue(clamp(newValue))
    if (newValue !== value && !isNaN(newValue)) {
      onChange(newValue)
    }
  }

  const clamp = (value: number) => {
    return Math.max(
      min ?? Number.NEGATIVE_INFINITY,
      Math.min(max ?? Number.POSITIVE_INFINITY, value),
    )
  }

  const onChangeInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    updateRawValue(clamp(parseFloat(e.target.value)))
  }

  const onBlur = () => {
    if (isNaN(rawValue)) {
      updateRawValue(min ?? 0)
    }
  }

  useEffect(() => {
    // Update the input value without triggering onChange
    setRawValue(value)
  }, [value])

  return (
    <input
      type="number"
      id={id}
      min={min}
      max={max}
      value={rawValue}
      step={step}
      onChange={onChangeInput}
      onKeyPress={onKeyPress}
      onBlur={onBlur}
      style={style}
      className={className}
    />
  )
}
