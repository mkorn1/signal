import { Range } from "../entities/geometry/Range"

export const filterEventsWithRange = <T extends { tick: number }>(
  events: T[],
  ...range: Range
): T[] => events.filter((e) => Range.contains(range, e.tick))

export const filterEventsWithScroll = <T extends { tick: number }>(
  events: T[],
  scrollLeftTick: number,
  widthTick: number,
): T[] =>
  filterEventsWithRange(events, ...Range.fromLength(scrollLeftTick, widthTick))

export const filterEventsOverlapRange = <
  T extends { tick: number; duration?: number },
>(
  events: T[],
  ...range: Range
): T[] => {
  return events.filter((e) => {
    if ("duration" in e && typeof e.duration === "number") {
      const eventTickEnd = e.tick + e.duration
      const noteRange = Range.create(e.tick, eventTickEnd)
      return Range.intersects(range, noteRange)
    }
    return Range.contains(range, e.tick)
  })
}

export const filterEventsOverlapScroll = <
  T extends { tick: number; duration?: number },
>(
  events: T[],
  scrollLeftTick: number,
  widthTick: number,
): T[] => {
  return filterEventsOverlapRange(
    events,
    ...Range.fromLength(scrollLeftTick, widthTick),
  )
}
