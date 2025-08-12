# app.py (parent-product bundle picker)
import os, time, json, textwrap
from typing import List, Optional, Dict
import requests
import streamlit as st
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# --- Streamlit Cloud secrets -> env ---
try:
    import streamlit as st  # already imported
    for k in ("SHOPIFY_STORE","SHOPIFY_TOKEN","GEMINI_API_KEY","API_VERSION","ADMIN_PASSWORD"):
        if k in st.secrets and not os.environ.get(k):
            os.environ[k] = str(st.secrets[k])
except Exception:
    pass

# --- Simple password gate ---
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.title("🔒 Summer Fridays Bundle Builder Login")
    pwd = st.text_input("Enter Password", type="password")
    if st.button("Enter"):
        if pwd and ADMIN_PASSWORD and pwd == ADMIN_PASSWORD:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Nope.")
    st.stop()

# =========================
# Config & setup
# =========================
load_dotenv()
SHOP = os.environ["SHOPIFY_STORE"]
TOKEN = os.environ["SHOPIFY_TOKEN"]
API_VERSION = os.environ.get("API_VERSION", "2025-07")
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

ADMIN_URL = f"https://{SHOP}/admin/api/{API_VERSION}/graphql.json"
HEADERS = {"Content-Type": "application/json", "X-Shopify-Access-Token": TOKEN}

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"

st.set_page_config(page_title="Shopify Bundle Builder (Parent Products)", page_icon="🧩", layout="wide")
st.title("🧩 Summer Fridays Bundle Builder — Parent Products (buyer picks variant)")

# =========================
# Models (use product_id, not variant_id)
# =========================
class ComponentDraft(BaseModel):
    product_id: str
    quantity: int = 1  # clamp >=1 later

class BundleDraft(BaseModel):
    title: str
    description_html: str
    price: float
    compare_at_price: Optional[float] = None
    components: List[ComponentDraft]

# =========================
# Gemini helpers (raw JSON; validate with Pydantic)
# =========================
def gemini_propose_bundle_draft(context: Dict) -> BundleDraft:
    """
    Ask Gemini for bundle title/desc/price and components using ONLY product_ids provided.
    """
    sys = textwrap.dedent("""
        You are a Shopify merchandising assistant.
        Return ONLY a single JSON object (no prose) with this schema:
        {
          "title": string,                  // <= 60 chars, specific
          "description_html": string,       // brief marketing copy (valid HTML; can include <ul><li>)
          "price": number,                  // non-negative; .00 or .99 if reasonable
          "compare_at_price": number|null,  // optional; >= price if discounting; else null
          "components": [                   // use provided product_ids; don't invent
            {"product_id": string, "quantity": number}
          ]
        }
        Rules:
        - Use ONLY product_ids from context; default quantity = 1 unless user asked otherwise.
        - Do not invent products/SKUs/IDs.
    """).strip()

    model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config={"response_mime_type": "application/json"}
    )

    prompt = (
        "Selected products and notes:\n"
        + json.dumps(context, indent=2)
        + "\nReturn ONLY the JSON object."
    )

    resp = model.generate_content([sys, prompt])
    data = json.loads(resp.text)
    return BundleDraft(**data)

def gemini_suggest_companions_products(anchor: Dict, candidates: List[Dict], k: int = 2) -> List[str]:
    """
    Ask Gemini to choose k companion product_ids from provided candidates based on complementarity.
    Returns a list of product_id strings (subset of the provided candidates).
    """
    sys = """
    You are a Shopify merchandising assistant. Choose complementary products
    to form a sensible fixed bundle around the anchor product.

    Return ONLY a JSON object like:
    {"product_ids": ["gid://shopify/Product/123", "gid://shopify/Product/456"]}

    Rules:
    - Use ONLY product_ids that appear in candidates.
    - Pick distinct product_ids; exclude the anchor's product_id.
    - Prefer complements (e.g., tint + blush + lip balm) and consider vendor/productType/tags/title.
    - Keep price mix reasonable for a bundle.
    """
    model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config={"response_mime_type": "application/json"}
    )

    anchor_compact = {
        "product_id": anchor["id"],
        "product_title": anchor["title"],
        "vendor": anchor.get("vendor"),
        "productType": anchor.get("productType"),
        "tags": anchor.get("tags"),
        "options": anchor.get("options"),
    }

    cand_compact = [{
        "product_id": c["id"],
        "product_title": c["title"],
        "vendor": c.get("vendor"),
        "productType": c.get("productType"),
        "tags": c.get("tags"),
        "options": c.get("options"),
    } for c in candidates[:200]]

    prompt = (
        f"anchor:\n{json.dumps(anchor_compact, indent=2)}\n\n"
        f"candidates:\n{json.dumps(cand_compact, indent=2)}\n\n"
        f"Pick exactly {k} companion product_ids from the candidates (exclude the anchor)."
    )
    resp = model.generate_content([sys, prompt])
    try:
        data = json.loads(resp.text)
        pids = data.get("product_ids") or []
        return [str(v) for v in pids][:k]
    except Exception:
        return []

# =========================
# Shopify GraphQL helpers (retries + pagination)
# =========================
def shopify_graphql(query: str, variables: dict = None, max_retries: int = 6):
    payload = {"query": query, "variables": variables or {}}
    backoff = 1.0
    for _ in range(max_retries):
        r = requests.post(ADMIN_URL, headers=HEADERS, json=payload, timeout=90)
        if r.status_code == 429:
            time.sleep(backoff); backoff = min(backoff * 2, 8); continue
        r.raise_for_status()
        data = r.json()
        if "errors" in data:
            if any("throttle" in (e.get("message","") + str(e)).lower() for e in data["errors"]):
                time.sleep(backoff); backoff = min(backoff * 2, 8); continue
            raise RuntimeError(data["errors"])
        return data["data"]
    raise RuntimeError("Shopify GraphQL: exceeded retry budget (throttled?)")

def fetch_all_products_only() -> list:
    """
    Fetch products (no variant paging needed). We need id/title/vendor/type/tags/options.
    """
    q = """
    query($first:Int!, $after:String){
      products(first:$first, after:$after, sortKey:TITLE) {
        edges {
          cursor
          node {
            id
            title
            vendor
            productType
            tags
            options { id name values }
          }
        }
        pageInfo { hasNextPage endCursor }
      }
    }
    """
    out, after = [], None
    while True:
        data = shopify_graphql(q, {"first": 250, "after": after})["products"]
        for e in data["edges"]:
            out.append(e["node"])
        if not data["pageInfo"]["hasNextPage"]:
            break
        after = data["pageInfo"]["endCursor"]
        time.sleep(0.25)
    return out

def build_component_inputs_from_products(selected_product_ids: list, prod_index: dict):
    """
    Build ProductBundleComponentInput[] from parent products.
    For each component, include optionSelections with ALL option values so the buyer can choose
    the variant on the bundle product page.
    """
    components = []
    for pid in selected_product_ids:
        p = prod_index[pid]
        optionSelections = []
        for opt in p.get("options", []) or []:
            values = [v for v in (opt.get("values") or []) if v]
            if not values:
                continue
            optionSelections.append({
                "componentOptionId": opt["id"],        # option id of the component product
                "name": opt["name"] or "Option",       # label shown on parent bundle
                "values": values                        # ALL allowed values so shopper can pick
            })
        components.append({
            "productId": pid,
            "quantity": 1,
            "optionSelections": optionSelections,       # [] if product has no options
        })
    return components

def product_bundle_create(title: str, components: List[dict]) -> str:
    mutation = """
    mutation($input: ProductBundleCreateInput!) {
      productBundleCreate(input: $input) {
        productBundleOperation { id status }
        userErrors { field message }
      }
    }
    """
    data = shopify_graphql(mutation, {"input": {"title": title, "components": components}})
    res = data["productBundleCreate"]
    if res["userErrors"]:
        raise RuntimeError(res["userErrors"])
    return res["productBundleOperation"]["id"]

def product_operation_poll(op_id: str, timeout_s=120) -> str:
    query = """
    query($id: ID!) {
      productOperation(id: $id) {
        __typename
        ... on ProductBundleOperation {
          status
          product { id }
          userErrors { message code }
        }
      }
    }
    """
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        op = shopify_graphql(query, {"id": op_id})["productOperation"]
        status = op.get("status")
        if status == "COMPLETE":
            prod = op.get("product")
            if not prod:
                raise RuntimeError("Bundle op COMPLETE but no product returned.")
            return prod["id"]
        elif status in ("ACTIVE", "CREATED"):
            time.sleep(2)
        else:
            raise RuntimeError(f"Bundle operation failed: {op}")
    raise TimeoutError("Timed out waiting for bundle creation to complete.")

def product_update(product_id: str, description_html: str, status: str = "ACTIVE"):
    mutation = """
    mutation($product: ProductUpdateInput!) {
      productUpdate(product: $product) {
        product { id status }
        userErrors { field message }
      }
    }
    """
    res = shopify_graphql(mutation, {"product": {"id": product_id, "status": status, "descriptionHtml": description_html}})
    if res["productUpdate"]["userErrors"]:
        raise RuntimeError(res["productUpdate"]["userErrors"])

def get_default_variant_id(product_id: str) -> str:
    query = """
    query($id: ID!){
      node(id:$id){
        ... on Product {
          variants(first:1){ nodes { id } }
        }
      }
    }
    """
    nodes = shopify_graphql(query, {"id": product_id})["node"]["variants"]["nodes"]
    if not nodes:
        raise RuntimeError("No variants on created bundle product.")
    return nodes[0]["id"]

def set_price_on_variant(product_id: str, variant_id: str, price: float, compare_at_price: Optional[float]):
    """
    Set price (and optional compareAtPrice) on the bundle's default variant.
    """
    mutation = """
    mutation($productId: ID!, $variants:[ProductVariantsBulkInput!]!) {
      productVariantsBulkUpdate(productId: $productId, variants: $variants) {
        productVariants { id }
        userErrors { field message }
      }
    }
    """
    v = {"id": variant_id, "price": f"{round(float(price),2):.2f}"}
    cap = None if compare_at_price in (None, "") else float(compare_at_price)
    if cap is not None and cap > 0:
        v["compareAtPrice"] = f"{round(cap,2):.2f}"
    res = shopify_graphql(mutation, {"productId": product_id, "variants": [v]})["productVariantsBulkUpdate"]
    if res["userErrors"]:
        raise RuntimeError(res["userErrors"])

def publish_to_current_channel(product_id: str):
    mutation = """
    mutation($id: ID!){
      publishablePublishToCurrentChannel(id:$id){
        publishable { ... on Product { id } }
        userErrors { field message }
      }
    }
    """
    res = shopify_graphql(mutation, {"id": product_id})["publishablePublishToCurrentChannel"]
    if res["userErrors"]:
        raise RuntimeError(res["userErrors"])

# Robust rerun helper
def _rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# =========================
# Load catalog once (PRODUCTS, not variants)
# =========================
if "catalog" not in st.session_state:
    with st.spinner("Loading products (first load may take a minute)…"):
        products = fetch_all_products_only()
        prod_index = {p["id"]: p for p in products}
        choices = [(f'{p["title"]} ({p["id"].split("/")[-1]})', p["id"]) for p in products]
        st.session_state.catalog = {"products": products, "prod_index": prod_index, "choices": choices}

catalog = st.session_state.catalog
product_labels = [label for label, _ in catalog["choices"]]
product_id_by_label = {label: pid for label, pid in catalog["choices"]}

# Session state
if "picked_product_ids" not in st.session_state:
    st.session_state.picked_product_ids = []
if "search_nonce" not in st.session_state:
    st.session_state.search_nonce = 0
if "last_suggested_pids" not in st.session_state:
    st.session_state.last_suggested_pids = []

# For Gemini context convenience
def _product_brief(pid: str):
    p = catalog["prod_index"][pid]
    return {
        "product_id": p["id"],
        "title": p["title"],
        "vendor": p.get("vendor"),
        "productType": p.get("productType"),
        "tags": p.get("tags"),
        "options": p.get("options"),
    }

# =========================
# UI
# =========================
left, right = st.columns([2, 1])

with left:
    # --- AI Suggest Companions (products) ---
    st.markdown("### 🔮 AI: suggest 2 companion products")
    st.caption("Pick one anchor product and let AI suggest two complementary products to add.")

    anchor_label = st.selectbox("Pick anchor product", options=["—"] + product_labels, index=0)
    col_suggest, col_add_suggested = st.columns([1,1])

    with col_suggest:
        disabled = (anchor_label == "—")
        if st.button("✨ Suggest 2 companions", use_container_width=True, disabled=disabled):
            anchor_pid = product_id_by_label[anchor_label]
            anchor_product = catalog["prod_index"][anchor_pid]

            # candidate pool: same vendor → same productType → others
            same_vendor, same_type, others = [], [], []
            avendor = (anchor_product.get("vendor") or "").lower()
            atype = (anchor_product.get("productType") or "").lower()
            for p in catalog["products"]:
                pid = p["id"]
                if pid == anchor_pid: 
                    continue
                vendor = (p.get("vendor") or "").lower()
                ptype = (p.get("productType") or "").lower()
                if avendor and vendor == avendor:
                    same_vendor.append(p)
                elif atype and ptype == atype:
                    same_type.append(p)
                else:
                    others.append(p)
            candidates = (same_vendor + same_type + others)[:400]

            try:
                pids = gemini_suggest_companions_products(anchor_product, candidates, k=2)
                # Validate & keep only product_ids we actually have
                valid = []
                for pid in pids:
                    if pid in catalog["prod_index"] and pid != anchor_pid and pid not in valid:
                        valid.append(pid)
                st.session_state.last_suggested_pids = valid
                if valid:
                    lbls = [f'{catalog["prod_index"][pid]["title"]} ({pid.split("/")[-1]})' for pid in valid]
                    st.success("Suggested:\n- " + "\n- ".join(lbls))
                else:
                    st.warning("No valid suggestions returned. Try a different anchor.")
            except Exception as e:
                st.error(f"Suggestion failed: {e}")

    with col_add_suggested:
        suggested = st.session_state.last_suggested_pids
        if st.button("➕ Add suggested to bundle", use_container_width=True, disabled=(len(suggested)==0)):
            for pid in suggested:
                if pid not in st.session_state.picked_product_ids:
                    st.session_state.picked_product_ids.append(pid)
            st.success("Added suggested products to your bundle.")

    st.markdown("---")

    # --- Search-as-you-type picker (PRODUCTS) ---
    st.markdown("### Search products")
    q = st.text_input(
        "Type to search by product title or ID",
        help="Filters locally. Try part of the product title or the numeric product ID."
    )

    def _label_matches(label: str, term: str) -> bool:
        t = (term or "").strip().lower()
        return True if not t else t in label.lower()

    filtered_labels = [l for l in product_labels if _label_matches(l, q)]

    multi_key = f"prod_multiselect_{st.session_state.search_nonce}"
    picked_labels = st.multiselect(
        "Search results",
        options=filtered_labels,
        key=multi_key,
        help="Pick one or more products, then click 'Add to bundle'."
    )

    col_add, col_clear = st.columns([1,1])
    with col_add:
        if st.button("➕ Add products to bundle", use_container_width=True, disabled=len(picked_labels) == 0):
            for lab in picked_labels:
                pid = product_id_by_label[lab]
                if pid not in st.session_state.picked_product_ids:
                    st.session_state.picked_product_ids.append(pid)
            st.session_state.search_nonce += 1
            _rerun()

    with col_clear:
        if st.button("🗑️ Clear all selected", use_container_width=True, disabled=len(st.session_state.picked_product_ids) == 0):
            st.session_state.picked_product_ids = []

    st.markdown("#### Your bundle components (products)")
    if not st.session_state.picked_product_ids:
        st.info("No components yet. Use AI suggestions or search above and add products.")
    else:
        remove_ids = []
        for pid in st.session_state.picked_product_ids:
            p = catalog["prod_index"][pid]
            label = f'{p["title"]} ({pid.split("/")[-1]})'
            c1, c2 = st.columns([6,1])
            c1.write(label)
            if c2.button("Remove", key=f"rm_{pid}"):
                remove_ids.append(pid)
        for pid in remove_ids:
            st.session_state.picked_product_ids.remove(pid)

    picked_product_ids = st.session_state.picked_product_ids

    # --- Gemini chat → draft (PRODUCTS) ---
    st.markdown("### Chat with AI about the bundle")
    st.caption("Ask for title, description, and pricing. The AI returns a structured draft you can tweak.")

    if "draft" not in st.session_state:
        st.session_state.draft = {
            "title": "",
            "description_html": "",
            "price": 0.0,
            "compare_at_price": None,
            "components": [{"product_id": pid, "quantity": 1} for pid in picked_product_ids],
        }

    user_msg = st.text_input("Your prompt (e.g., “Tint + Blush + Lip Balm summer set at 15% off.”)")
    colA, colB = st.columns([1,1])

    with colA:
        if st.button("💡 Propose details with Gemini", use_container_width=True, disabled=not picked_product_ids):
            context = {
                "selected_products": [_product_brief(pid) for pid in picked_product_ids],
                "notes": user_msg or "",
            }
            try:
                bd = gemini_propose_bundle_draft(context)
                draft = bd.dict()

                # ---- Server-side sanity checks ----
                draft["price"] = max(0.0, float(draft.get("price") or 0.0))
                cap = draft.get("compare_at_price")
                draft["compare_at_price"] = float(cap) if cap not in (None, "") else None
                for c in draft.get("components", []):
                    c["quantity"] = max(1, int(c.get("quantity") or 1))

                # Ensure components exist and belong to picked set
                if not draft.get("components"):
                    draft["components"] = [{"product_id": pid, "quantity": 1} for pid in picked_product_ids]
                else:
                    # keep only components that are in picked_product_ids
                    draft["components"] = [
                        c for c in draft["components"] if c.get("product_id") in picked_product_ids
                    ] or [{"product_id": pid, "quantity": 1} for pid in picked_product_ids]

                st.session_state.draft = draft
                st.success("Draft received from AI.")
            except Exception as e:
                st.error(f"Gemini failed to propose draft: {e}")

    with colB:
        if st.button("🧹 Clear draft", use_container_width=True):
            st.session_state.draft = {
                "title": "",
                "description_html": "",
                "price": 0.0,
                "compare_at_price": None,
                "components": [{"product_id": pid, "quantity": 1} for pid in picked_product_ids],
            }

    # Editable draft
    d = st.session_state.draft
    d["title"] = st.text_input("Title", d.get("title",""))
    d["description_html"] = st.text_area("Description (HTML allowed)", d.get("description_html",""), height=160)
    d["price"] = st.number_input("Bundle price", value=float(d.get("price") or 0.0), step=0.5, min_value=0.0)
    d["compare_at_price"] = st.number_input("Compare-at price (optional)", value=float(d.get("compare_at_price") or 0.0), step=0.5, min_value=0.0)

    # Keep components aligned with current picks unless user customized quantities already
    existing_ids = {c.get("product_id") for c in d.get("components", []) if c.get("product_id")}
    for pid in picked_product_ids:
        if pid not in existing_ids:
            d.setdefault("components", []).append({"product_id": pid, "quantity": 1})
    d["components"] = [c for c in d.get("components", []) if c.get("product_id") in picked_product_ids]
    st.session_state.draft = d

with right:
    st.subheader("Create bundle")
    st.caption("Creates a componentized product from PARENT products (buyer chooses variant), sets description & price, and publishes.")
    if st.button("🚀 Create bundle on Shopify", type="primary", use_container_width=True, disabled=len(st.session_state.picked_product_ids) == 0):
        try:
            draft: Dict = st.session_state["draft"]

            comp_inputs = build_component_inputs_from_products(
                [c["product_id"] for c in draft.get("components", []) if c.get("product_id")],
                catalog["prod_index"]
            )

            # 1) Create bundle (returns operation id)
            op_id = product_bundle_create(draft["title"], comp_inputs)

            # 2) Poll until created, get product id
            product_id = product_operation_poll(op_id)

            # 3) Update description & set ACTIVE
            product_update(product_id, draft.get("description_html",""), status="ACTIVE")

            # 4) Set price on default variant (requires productId)
            default_vid = get_default_variant_id(product_id)
            price = float(draft.get("price") or 0.0)
            cap = draft.get("compare_at_price")
            cap = float(cap) if cap not in (None, "") else None
            set_price_on_variant(product_id, default_vid, price, cap)

            # 5) Publish to current channel
            publish_to_current_channel(product_id)

            st.success("Bundle created from parent products & published! Shoppers can now choose variants on the bundle page.")
            admin_link = f"https://{SHOP}/admin/products/{product_id.split('/')[-1]}"
            st.markdown(f"[Open in Shopify Admin]({admin_link})")
        except Exception as e:
            st.error(f"Failed to create bundle: {e}")
