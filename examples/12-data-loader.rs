use dfdx::data::*;
use dfdx::prelude::*;

fn main() {
    let dataset = vec![
        tensor([1.0, 2.0]),
        tensor([3.0, 4.0]),
        tensor([5.0, 6.0]),
        tensor([7.0, 8.0]),
        tensor([9.0, 10.0]),
    ];

    let loader = Builder::new(dataset)
        .batch_size(2)
    ;

    for batch in &loader {
        dbg!(batch);
    }


}