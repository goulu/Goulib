import { Host, Component, State, h } from '@stencil/core';

@Component({
  tag: 'app-home',
  styleUrl: 'app-home.css'
})
export class AppHome {

  @State() data: any = {}

  async parse(filename: string) {
    const response = await fetch(filename)
    let text = await response.text()
    let dups = []
    for (const line of text.split("\r\n")) {
      if (line.length < 3) {
        dups.map((dup) => {
          let n = dup.lastIndexOf("\\")
          let dir = dup.slice(0, n - 1)
          this.data[dir] = [...(this.data[dir] || []), dups]
        })
        dups = []
      } else {
        dups.push(line)
      }
    }
    console.log(this.data)
  }

  componentWillLoad() {
    this.parse('./assets/dups.txt')
  }

  // https://www.freakyjolly.com/ionic-4-implement-infinite-scroll-list-with-virtual-scroll-list-in-ionic-4-application/#.Xv4NgSiFqCo

  render() {
    console.log('render')
    return (
      <ion-virtual-scroll>
        {Object.keys(this.data).map((key) =>
          <ion-item>
            <ion-label>{key}</ion-label>
          </ion-item>)}
      </ion-virtual-scroll>
    )
  }
}
