use crate::segments::Segment;
use anyhow::Result;

pub trait SegmentEncoder {
    fn write_segment(&mut self, seg: &Segment) -> Result<()>;
    fn close(&mut self) -> Result<()>;
}
