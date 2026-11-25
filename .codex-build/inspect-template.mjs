import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';

const p = await PresentationFile.importPptx(await FileBlob.load('C:/Users/USER/Desktop/26C02_概率与概率图模型.pptx'));
const q = await p.inspect({select:'$.slides[*]',include:['id','index','layout','title','elements.id','elements.kind','elements.text','elements.position','elements.style']});
console.log(q.ndjson);
await fs.mkdir('C:/Users/USER/Desktop/nlp/.codex-build/import-render',{recursive:true});
for (let i=0;i<p.slides.items.length;i++) {
  const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});
  await fs.writeFile(`C:/Users/USER/Desktop/nlp/.codex-build/import-render/${i+1}.png`,new Uint8Array(await b.arrayBuffer()));
}
