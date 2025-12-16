import { FC } from "react"
import { Localized } from "../../localize/useLocalization"
import { EditMenu } from "./EditMenu"
import { NavButton } from "./Navigation"

export const EditMenuButton: FC = () => {
  return (
    <EditMenu
      trigger={
        <NavButton id="tab-edit">
          <span>
            <Localized name="edit" />
          </span>
        </NavButton>
      }
    />
  )
}
