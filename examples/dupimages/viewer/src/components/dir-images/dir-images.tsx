import { Component, Host, h } from '@stencil/core';

@Component({
  tag: 'dir-images',
  styleUrl: 'dir-images.css',
  shadow: true,
})
export class DirImages {

  render() {
    return (
      <Host>
        <slot></slot>
      </Host>
    );
  }

}
