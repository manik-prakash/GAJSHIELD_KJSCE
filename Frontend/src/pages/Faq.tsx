import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

export default function Faq() {
  return (
    <div className="flex flex-col pl-10 pr-10 pb-10 bg-[#000000] text-white">
      <Accordion type="single" collapsible>
        <AccordionItem value="item-1">
          <AccordionTrigger className="text-xl">
            How do malware detection tools work?
          </AccordionTrigger>
          <AccordionContent className="text-lg">
            They use signature matching, heuristic analysis, behavior
            monitoring, and machine learning to identify threats.
          </AccordionContent>
        </AccordionItem>
      </Accordion>

      <Accordion type="single" collapsible>
        <AccordionItem value="item-1">
          <AccordionTrigger className="text-xl">
          What's the difference between real-time and on-demand scanning?
          </AccordionTrigger>
          <AccordionContent className="text-lg">
          Real-time scanning checks files continuously as they're accessed; on-demand scanning only runs when manually started.
          </AccordionContent>
        </AccordionItem>
      </Accordion>

      <Accordion type="single" collapsible>
        <AccordionItem value="item-1">
          <AccordionTrigger className="text-xl">
          Do I need a malware detection tool if my OS has built-in protection?
          </AccordionTrigger>
          <AccordionContent className="text-lg">
          Built-in protection offers basic security, but dedicated tools typically provide more comprehensive features and stronger protection against advanced threats.
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
}
