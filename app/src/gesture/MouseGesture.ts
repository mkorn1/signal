export interface MouseGesture<Params extends any[] = []> {
  onMouseDown(e: MouseEvent, ...params: Params): void
  onMouseMove?(e: MouseEvent): void
  onMouseUp?(e: MouseEvent): void
}
