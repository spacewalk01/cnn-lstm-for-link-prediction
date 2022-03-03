## BiLSTM CNN for knowledge graph completion in Keras


Keras implementation for the paper titled "Path-based reasoning approach for knowledge graph completion using CNN-BiLSTM with attention mechanism". Given a candidate relation and two entities, it encodes paths that connect the entities into a low-dimensional space using a convolutional operation followed by BiLSTM. An attention layer is also applied to capture the semantic correlation between a candidate relation and each path between two entities and attentively extract reasoning evidence from the representation of multiple paths to predict whether the entities should be connected by the candidate relation. 
<p align="center">
<img src = "1.JPG" width="50%" height="50%" class="center">
</p>
### Required Files

- data/processed_data.tar.gz - dataset files containing grounded paths with relations and entities (e.g e1, r1, e2, r2, e3).
- tasks - can be downloaded from [1].

For generating relation paths such as (r1, r2, ..., rk), we used [2].

The original knowledge graph data we used for our experiments can be found in [1] which includes tasks datasets with train/test triples.

### Reference:
- [1] https://github.com/shehzaadzd/MINERVA/tree/master/datasets/data_preprocessed
- [2] https://github.com/noon99jaki/pra

