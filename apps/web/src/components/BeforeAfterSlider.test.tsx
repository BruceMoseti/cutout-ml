import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { BeforeAfterSlider } from './BeforeAfterSlider';

/**
 * The slider is the primary control on the studio page, and the failure mode that matters
 * is that it becomes mouse-only. These tests pin the keyboard contract, which is the part
 * a refactor silently breaks.
 */
describe('BeforeAfterSlider', () => {
  const props = { beforeSrc: 'blob:before', afterSrc: 'blob:after' };

  it('exposes a slider with a mid-point default', () => {
    render(<BeforeAfterSlider {...props} />);
    const slider = screen.getByRole('slider', { name: /comparison position/i });
    expect(slider).toHaveAttribute('aria-valuenow', '50');
    expect(slider).toHaveAttribute('aria-valuemin', '0');
    expect(slider).toHaveAttribute('aria-valuemax', '100');
  });

  it('moves by 2 with an arrow key and 10 with shift held', async () => {
    const user = userEvent.setup();
    render(<BeforeAfterSlider {...props} />);
    const slider = screen.getByRole('slider', { name: /comparison position/i });

    slider.focus();
    await user.keyboard('{ArrowRight}');
    expect(slider).toHaveAttribute('aria-valuenow', '52');

    await user.keyboard('{ArrowLeft}{ArrowLeft}');
    expect(slider).toHaveAttribute('aria-valuenow', '48');

    await user.keyboard('{Shift>}{ArrowRight}{/Shift}');
    expect(slider).toHaveAttribute('aria-valuenow', '58');
  });

  it('clamps at both ends via Home and End', async () => {
    const user = userEvent.setup();
    render(<BeforeAfterSlider {...props} />);
    const slider = screen.getByRole('slider', { name: /comparison position/i });

    slider.focus();
    await user.keyboard('{Home}');
    expect(slider).toHaveAttribute('aria-valuenow', '0');
    await user.keyboard('{ArrowLeft}');
    expect(slider).toHaveAttribute('aria-valuenow', '0');

    await user.keyboard('{End}');
    expect(slider).toHaveAttribute('aria-valuenow', '100');
    await user.keyboard('{ArrowRight}');
    expect(slider).toHaveAttribute('aria-valuenow', '100');
  });

  it('renders both images with distinguishable alt text', () => {
    render(<BeforeAfterSlider {...props} alt="Cut out result" />);
    expect(screen.getByAltText('Cut out result (original)')).toBeInTheDocument();
    expect(screen.getByAltText('Cut out result')).toBeInTheDocument();
  });
});
