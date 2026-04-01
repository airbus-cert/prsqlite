// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

mod btree;
mod cursor;
mod expression;
mod header;
mod pager;
mod parser;
mod payload;
mod query;
mod record;
mod schema;
#[cfg(test)]
pub mod test_utils;
mod token;
mod utils;
mod value;

use std::cell::Cell;
use std::cell::RefCell;
use std::fmt::Display;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::panic::RefUnwindSafe;
// use std::os::unix::fs::FileExt;
use std::path::Path;

use anyhow::bail;
use anyhow::Context;
use btree::BtreeContext;
use cursor::BtreeCursor;
use expression::DataContext;
use expression::Expression;
use header::DatabaseHeader;
use header::DatabaseHeaderMut;
use header::DATABASE_HEADER_SIZE;
use pager::PageId;
use pager::Pager;
use pager::PAGE_ID_1;
use parser::expect_no_more_token;
use parser::expect_semicolon;
use parser::parse_sql;
use parser::Delete;
use parser::Insert;
use parser::Parser;
use parser::ResultColumn;
use parser::Select;
use parser::Stmt;
use query::Query;
use query::QueryPlan;
use query::RowData;
use record::RecordPayload;
use schema::ColumnNumber;
use schema::Index;
use schema::Schema;
use schema::Table;
pub use value::Buffer;
use value::Collation;
use value::TypeAffinity;
pub use value::Value;
use value::ValueCmp;
use value::DEFAULT_COLLATION;

// Original SQLite support both 32-bit or 64-bit rowid. prsqlite only support
// 64-bit rowid.
const MAX_ROWID: i64 = i64::MAX;

#[derive(Debug)]
pub enum Error<'a> {
    Parse(parser::Error<'a>),
    Pager(pager::Error),
    Cursor(cursor::Error),
    Expression(expression::Error),
    Query(query::Error),
    UniqueConstraintViolation,
    DataTypeMismatch,
    Unsupported(&'static str),
    Other(anyhow::Error),
}

impl<'a> From<parser::Error<'a>> for Error<'a> {
    fn from(e: parser::Error<'a>) -> Self {
        Self::Parse(e)
    }
}

impl From<cursor::Error> for Error<'_> {
    fn from(e: cursor::Error) -> Self {
        Self::Cursor(e)
    }
}

impl From<expression::Error> for Error<'_> {
    fn from(e: expression::Error) -> Self {
        Self::Expression(e)
    }
}

impl From<query::Error> for Error<'_> {
    fn from(e: query::Error) -> Self {
        Self::Query(e)
    }
}

impl From<anyhow::Error> for Error<'_> {
    fn from(e: anyhow::Error) -> Self {
        Self::Other(e)
    }
}

impl std::error::Error for Error<'_> {}

impl Display for Error<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Parse(e) => {
                write!(f, "SQL parser error: {}", e)
            }
            Error::Cursor(e) => {
                write!(f, "Btree cursor error: {}", e)
            }
            Error::Expression(e) => {
                write!(f, "expression error: {}", e)
            }
            Error::Query(e) => {
                write!(f, "query error: {}", e)
            }
            Error::DataTypeMismatch => {
                write!(f, "data type mismatch")
            }
            Error::UniqueConstraintViolation => {
                write!(f, "unique constraint violation")
            }
            Error::Unsupported(msg) => {
                write!(f, "unsupported: {}", msg)
            }
            Error::Pager(msg) => {
                write!(f, "pager: {}", msg)
            }
            Error::Other(e) => write!(f, "{}", e),
        }
    }
}

pub type Result<'a, T> = std::result::Result<T, Error<'a>>;

pub trait ReadExactAt {
    fn read_exact_at(&self, buf: &mut [u8], at: u64) -> std::result::Result<(), pager::Error>;
}

impl ReadExactAt for File {
    fn read_exact_at(&self, buf: &mut [u8], at: u64) -> std::result::Result<(), pager::Error> {
        let mut t = self.clone();
        t.seek(SeekFrom::Start(at))?;
        Ok(t.read_exact(buf)?)
    }
}

impl ReadExactAt for Vec<u8> {
    fn read_exact_at(&self, buf: &mut [u8], at: u64) -> std::result::Result<(), pager::Error> {
        let at_size = at as usize;
        if at_size + buf.len() > self.len() {
            return Err(pager::Error::NoSpace)
        }
        buf.copy_from_slice(&self[at_size..at_size + buf.len()]);
        Ok(())
    }
}

pub struct Connection<T: ReadExactAt> {
    pager: Pager<T>,
    btree_ctx: BtreeContext,
    schema: RefCell<Option<Schema>>,
    /// Number of running read or write.
    ///
    /// \> 0 : read(s) running
    /// 0   : no read/write
    /// -1  : write running
    ref_count: Cell<i64>,
}

impl Connection<File> {
    pub fn from_file(filename: &Path) -> anyhow::Result<Self> {
        // TODO: support read only mode.
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(filename)
            .with_context(|| format!("failed to open file: {:?}", filename))?;
        Self::from_reader(file)
    }
}

impl<T: ReadExactAt> Connection<T> {

    pub fn from_reader(mut reader: T) -> anyhow::Result<Self> {
        let mut buf = [0; DATABASE_HEADER_SIZE];
        reader.read_exact_at(&mut buf, 0)?;
        let header = DatabaseHeader::from(&buf);
        header
            .validate()
            .map_err(|e| anyhow::anyhow!("database header invalid: {e}"))?;
        let pagesize = header.pagesize();
        // pagesize is bigger than or equal to 512.
        // reserved is smaller than or equal to 255.
        let usable_size = pagesize - header.reserved() as u32;
        let pager = Pager::new(
            reader,
            header.n_pages(),
            pagesize,
            usable_size,
            header.first_freelist_trunk_page_id(),
            header.n_freelist_pages(),
        )?;
        Ok(Self {
            pager,
            btree_ctx: BtreeContext::new(usable_size),
            schema: RefCell::new(None),
            ref_count: Cell::new(0),
        })
    }

    pub fn prepare<'a, 'conn>(&'conn self, sql: &'a str) -> Result<'a, Statement<'conn, T>> {
        let input = sql.as_bytes();
        let mut parser = Parser::new(input);
        let statement = parse_sql(&mut parser)?;
        expect_semicolon(&mut parser)?;
        expect_no_more_token(&parser)?;

        match statement {
            Stmt::Select(select) => Ok(Statement::Query(self.prepare_select(select)?))
        }
    }

    fn load_schema(&self) -> anyhow::Result<()> {
        let schema_table = Schema::schema_table();
        let columns = schema_table
            .get_all_columns()
            .map(Expression::Column)
            .collect::<Vec<_>>();
        *self.schema.borrow_mut() = Some(Schema::generate(
            SelectStatement::new(
                self,
                schema_table.root_page_id,
                columns,
                Expression::one(),
                QueryPlan::FullScan,
            ),
            schema_table,
        )?);
        Ok(())
    }

    fn prepare_select<'a>(&self, select: Select<'a>) -> Result<'a, SelectStatement<T>> {
        if self.schema.borrow().is_none() {
            self.load_schema()?;
        }
        let schema_cell = self.schema.borrow();
        let schema = schema_cell.as_ref().unwrap();
        let table_name = select.table_name.dequote();
        let table = schema.get_table(&table_name).ok_or(anyhow::anyhow!(
            "table not found: {:?}",
            std::str::from_utf8(&table_name).unwrap_or_default()
        ))?;

        let mut columns = Vec::new();
        for column in select.columns {
            match column {
                ResultColumn::All => {
                    columns.extend(table.get_all_columns().map(Expression::Column));
                }
                ResultColumn::Expr((expr, _alias)) => {
                    // TODO: consider alias.
                    columns.push(Expression::from(expr, Some(table))?);
                }
                ResultColumn::AllOfTable(_table_name) => {
                    todo!("ResultColumn::AllOfTable");
                }
            }
        }

        let filter = select
            .filter
            .map(|expr| Expression::from(expr, Some(table)))
            .transpose()?
            .unwrap_or(Expression::one());

        let query_plan = QueryPlan::generate(table, &filter);

        Ok(SelectStatement::new(
            self,
            table.root_page_id,
            columns,
            filter,
            query_plan,
        ))
    }

    fn start_read(&self) -> anyhow::Result<ReadTransaction<T>> {
        // TODO: Lock across processes
        let ref_count = self.ref_count.get();
        if ref_count >= 0 {
            self.ref_count.set(ref_count + 1);
            Ok(ReadTransaction(self))
        } else {
            bail!("write statment running");
        }
    }
}

struct ReadTransaction<'a, T: ReadExactAt>(&'a Connection<T>);

impl<T: ReadExactAt> Drop for ReadTransaction<'_, T> {
    fn drop(&mut self) {
        self.0.ref_count.set(self.0.ref_count.get() - 1);
    }
}

pub trait ExecutionStatement {
    fn execute(&self) -> Result<u64>;
}

pub enum Statement<'conn, T: ReadExactAt> {
    Query(SelectStatement<'conn, T>),
    Execution(Box<dyn ExecutionStatement + 'conn>),
}

impl<'conn, T: ReadExactAt> Statement<'conn, T> {
    pub fn query(&'conn self) -> anyhow::Result<Rows<'conn, T>> {
        match self {
            Self::Query(stmt) => stmt.query(),
            Self::Execution(_) => bail!("execute statement not support query"),
        }
    }

    pub fn execute(&'conn self) -> Result<u64> {
        match self {
            Self::Query(_) => Err(Error::Unsupported("select statement not support execute")),
            Self::Execution(stmt) => stmt.execute(),
        }
    }
}

pub struct SelectStatement<'conn, T: ReadExactAt> {
    conn: &'conn Connection<T>,
    table_page_id: PageId,
    columns: Vec<Expression>,
    filter: Expression,
    query_plan: QueryPlan,
}

impl<'conn, T : ReadExactAt> SelectStatement<'conn, T> {
    pub(crate) fn new(
        conn: &'conn Connection<T>,
        table_page_id: PageId,
        columns: Vec<Expression>,
        filter: Expression,
        query_plan: QueryPlan,
    ) -> Self {
        Self {
            conn,
            table_page_id,
            columns,
            filter,
            query_plan,
        }
    }

    pub fn query(&'conn self) -> anyhow::Result<Rows<'conn, T>> {
        let read_txn = self.conn.start_read()?;
        // TODO: check schema version.

        let query = Query::new(
            self.table_page_id,
            &self.conn.pager,
            &self.conn.btree_ctx,
            &self.query_plan,
            &self.filter,
        )?;

        Ok(Rows {
            _read_txn: read_txn,
            stmt: self,
            query,
        })
    }
}

pub struct Rows<'conn, T: ReadExactAt> {
    _read_txn: ReadTransaction<'conn, T>,
    stmt: &'conn SelectStatement<'conn, T>,
    query: Query<'conn, T>,
}

impl<'conn, T: ReadExactAt> Rows<'conn, T> {
    pub fn next_row(&mut self) -> Result<Option<Row<'_, T>>> {
        if let Some(data) = self.query.next()? {
            Ok(Some(Row {
                stmt: self.stmt,
                data,
            }))
        } else {
            Ok(None)
        }
    }
}

pub struct Row<'a, T: ReadExactAt> {
    stmt: &'a SelectStatement<'a, T>,
    data: RowData<'a, T>,
}

impl<'a, T: ReadExactAt> Row<'a, T> {
    pub fn parse(&self) -> Result<Columns<'_>> {
        let mut columns = Vec::with_capacity(self.stmt.columns.len());
        for expr in self.stmt.columns.iter() {
            let (value, _, _) = expr.execute(Some(&self.data))?;
            columns.push(value);
        }
        Ok(Columns(columns))
    }
}

pub struct Columns<'a>(Vec<Option<Value<'a>>>);

impl<'a> Columns<'a> {
    pub fn get(&self, i: usize) -> Option<&Value<'a>> {
        if let Some(Some(v)) = self.0.get(i) {
            Some(v)
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Option<Value<'a>>> {
        self.0.iter()
    }
}

struct InsertRecord {
    rowid: Option<Expression>,
    columns: Vec<(Expression, TypeAffinity)>,
}

struct IndexSchema {
    root_page_id: PageId,
    columns: Vec<(ColumnNumber, Collation)>,
}

impl IndexSchema {
    fn create(table: &Table, index: &Index) -> Self {
        let mut columns = index
            .columns
            .iter()
            .map(|column_number| {
                let collation = if let ColumnNumber::Column(column_idx) = column_number {
                    &table.columns[*column_idx].collation
                } else {
                    &DEFAULT_COLLATION
                };
                (*column_number, collation.clone())
            })
            .collect::<Vec<_>>();
        columns.push((ColumnNumber::RowId, DEFAULT_COLLATION.clone()));

        IndexSchema {
            root_page_id: index.root_page_id,
            columns,
        }
    }
}
