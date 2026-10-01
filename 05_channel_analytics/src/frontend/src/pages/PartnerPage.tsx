import { Anchor, Box } from "grommet";
import { FormPrevious } from "grommet-icons";
import { useNavigate, useParams } from "react-router-dom";
import { PartnerContent } from "../components/PartnerPanel";

export function PartnerPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  return (
    <Box gap="medium">
      <Anchor icon={<FormPrevious />} label="Back" onClick={() => (window.history.length > 1 ? navigate(-1) : navigate("/at-risk"))} />
      <PartnerContent id={id} />
    </Box>
  );
}
