import * as React from "react"
import * as SeparatorPrimitive from "@radix-ui/react-separator"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"

const Separator = React.forwardRef<
  React.ElementRef<typeof SeparatorPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SeparatorPrimitive.Root> & {
    animated?: boolean
  }
>(
  (
    { className, orientation = "horizontal", decorative = true, animated = false, ...props },
    ref
  ) => (
    <SeparatorPrimitive.Root
      ref={ref}
      decorative={decorative}
      orientation={orientation}
      className={cn(
        "shrink-0 bg-border",
        orientation === "horizontal" ? "h-[1px] w-full" : "h-full w-[1px]",
        className
      )}
      {...props}
    >
      {animated && (
        <motion.div
          className="h-full w-full bg-gradient-to-r from-transparent via-current to-transparent opacity-50"
          animate={{
            x: orientation === "horizontal" ? ["-100%", "100%"] : 0,
            y: orientation === "vertical" ? ["-100%", "100%"] : 0,
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: "linear",
          }}
        />
      )}
    </SeparatorPrimitive.Root>
  )
)
Separator.displayName = SeparatorPrimitive.Root.displayName

export { Separator }