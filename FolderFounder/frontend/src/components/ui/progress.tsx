import * as React from "react"
import * as ProgressPrimitive from "@radix-ui/react-progress"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"

const Progress = React.forwardRef<
  React.ElementRef<typeof ProgressPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof ProgressPrimitive.Root> & {
    indicatorClassName?: string
    showPercentage?: boolean
  }
>(({ className, value, indicatorClassName, showPercentage = false, ...props }, ref) => (
  <div className="relative">
    <ProgressPrimitive.Root
      ref={ref}
      className={cn(
        "relative h-4 w-full overflow-hidden rounded-full bg-secondary",
        "bg-gray-200 dark:bg-gray-800",
        className
      )}
      {...props}
    >
      <ProgressPrimitive.Indicator asChild>
        <motion.div
          className={cn(
            "h-full bg-gradient-to-r from-blue-500 to-purple-600",
            "shadow-lg shadow-blue-500/25",
            indicatorClassName
          )}
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{
            type: "spring",
            stiffness: 100,
            damping: 20,
            mass: 1
          }}
        >
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
            animate={{ x: ["-100%", "100%"] }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          />
        </motion.div>
      </ProgressPrimitive.Indicator>
    </ProgressPrimitive.Root>
    {showPercentage && (
      <motion.span
        className="absolute right-2 top-1/2 -translate-y-1/2 text-xs font-medium text-white mix-blend-difference"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      >
        {value}%
      </motion.span>
    )}
  </div>
))
Progress.displayName = ProgressPrimitive.Root.displayName

export { Progress }