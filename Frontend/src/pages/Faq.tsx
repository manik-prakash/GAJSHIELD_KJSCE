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
          <AccordionTrigger className="text-xl">Is it accessible?</AccordionTrigger>
          <AccordionContent className="text-lg">
            Yes. It adheres to the WAI-ARIA design pattern.
          </AccordionContent>
        </AccordionItem>
      </Accordion>

      <Accordion type="single" collapsible >
        <AccordionItem value="item-1">
          <AccordionTrigger className="text-xl">Is it accessible?</AccordionTrigger>
          <AccordionContent className="text-lg">
            Yes. It adheres to the WAI-ARIA design pattern.
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
}
